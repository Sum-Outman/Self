/**
 * 状态管理stores索引文件
 * 导出所有store和相关的钩子函数
 */

// UI状态管理
export * from './ui.store';
export {
  useSidebar,
  useMobileSidebar,
  useLoading,
  useMessages,
  useConfirmDialog,
  useModal,
} from './ui.store';

// 训练状态管理
export * from './training.store';
export {
  useTrainingStatus,
  useTrainingProgress,
  useTrainingConfig,
  useTrainingControls,
  useTrainingMetrics,
  useCheckpoints,
} from './training.store';

// 用户偏好管理
export * from './userPreferences.store';
export {
  useThemePreference,
  useLayoutPreferences,
  useNotificationPreferences,
  useAccessibilityPreferences,
  useDataPreferences,
  usePrivacyPreferences,
  useShortcuts,
  useFeaturePreferences,
  useCustomSettings,
  resetUserPreferences,
  exportUserPreferences,
  importUserPreferences,
} from './userPreferences.store';

// 通用store工具函数
import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';

/**
 * 创建持久化store的辅助函数
 * @param name store名称（用于localStorage key）
 * @param initialState 初始状态
 * @param partialize 选择持久化的状态字段
 */
export function createPersistedStore<T extends object>(
  name: string,
  initialState: T,
  partialize?: (state: T) => Partial<T>
) {
  return create<T>()(
    persist(
      () => initialState,
      {
        name,
        storage: createJSONStorage(() => localStorage),
        partialize,
      }
    )
  );
}

/**
 * 创建带有操作的store的辅助函数
 * @param initialState 初始状态
 * @param actions store操作方法
 */
export function createStoreWithActions<
  TState extends object,
  TActions extends Record<string, (...args: any[]) => void>
>(
  initialState: TState,
  actions: (set: any, get: any) => TActions
) {
  return create<TState & TActions>()((set, get) => ({
    ...initialState,
    ...actions(set, get),
  }));
}

/**
 * 创建模块化store的辅助函数
 * @param moduleName 模块名称
 * @param initialState 初始状态
 * @param actions 操作方法
 */
export function createModuleStore<
  TState extends object,
  TActions extends Record<string, (...args: any[]) => void>
>(
  moduleName: string,
  initialState: TState,
  actions: (set: any, get: any) => TActions
) {
  return createPersistedStore(
    `${moduleName}-store`,
    {
      ...initialState,
      ...actions(() => {}, () => ({})), // 临时函数，实际在create中会被替换
    },
    (state) => state
  );
}

// Store管理器
export class StoreManager {
  private static instance: StoreManager;
  private stores: Map<string, any> = new Map();
  
  static getInstance(): StoreManager {
    if (!StoreManager.instance) {
      StoreManager.instance = new StoreManager();
    }
    return StoreManager.instance;
  }
  
  registerStore(name: string, store: any) {
    this.stores.set(name, store);
  }
  
  getStore<T>(name: string): T | undefined {
    return this.stores.get(name) as T;
  }
  
  resetStore(name: string) {
    const store = this.stores.get(name);
    if (store && typeof store.getState === 'function') {
      const initialState = store.getState();
      if (typeof initialState.reset === 'function') {
        initialState.reset();
      }
    }
  }
  
  resetAllStores() {
    for (const [name, _store] of this.stores) {
      this.resetStore(name);
    }
  }
  
  getStoreState(name: string) {
    const store = this.stores.get(name);
    if (store && typeof store.getState === 'function') {
      return store.getState();
    }
    return null;
  }
}

// 导出store管理器实例
export const storeManager = StoreManager.getInstance();