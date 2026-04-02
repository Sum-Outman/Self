/**
 * UI状态管理store
 * 使用zustand管理全局UI状态
 */
import { create } from 'zustand';
import { persist } from 'zustand/middleware';

// UI状态类型定义
export interface UIState {
  // 侧边栏状态
  sidebarOpen: boolean;
  // 移动端侧边栏状态
  mobileSidebarOpen: boolean;
  // 当前活动页面
  activePage: string;
  // 通知面板状态
  notificationsOpen: boolean;
  // 搜索面板状态
  searchOpen: boolean;
  // 设置面板状态
  settingsOpen: boolean;
  // 加载状态
  isLoading: boolean;
  // 全局加载文本
  loadingText: string;
  // 错误消息
  errorMessage: string | null;
  // 成功消息
  successMessage: string | null;
  // 警告消息
  warningMessage: string | null;
  // 确认对话框状态
  confirmDialog: {
    open: boolean;
    title: string;
    message: string;
    onConfirm: () => void;
    onCancel: () => void;
  };
  // 模态框状态
  modals: Record<string, boolean>;
  
  // 操作方法
  toggleSidebar: () => void;
  toggleMobileSidebar: () => void;
  setActivePage: (page: string) => void;
  toggleNotifications: () => void;
  toggleSearch: () => void;
  toggleSettings: () => void;
  setLoading: (loading: boolean, text?: string) => void;
  setError: (message: string | null) => void;
  setSuccess: (message: string | null) => void;
  setWarning: (message: string | null) => void;
  showConfirmDialog: (
    title: string, 
    message: string, 
    onConfirm: () => void, 
    onCancel?: () => void
  ) => void;
  hideConfirmDialog: () => void;
  openModal: (modalId: string) => void;
  closeModal: (modalId: string) => void;
  toggleModal: (modalId: string) => void;
  resetUI: () => void;
}

// 初始状态
const initialState = {
  sidebarOpen: true,
  mobileSidebarOpen: false,
  activePage: 'dashboard',
  notificationsOpen: false,
  searchOpen: false,
  settingsOpen: false,
  isLoading: false,
  loadingText: '加载中...',
  errorMessage: null,
  successMessage: null,
  warningMessage: null,
  confirmDialog: {
    open: false,
    title: '',
    message: '',
    onConfirm: () => {},
    onCancel: () => {},
  },
  modals: {},
};

// 创建UI store
export const useUIStore = create<UIState>()(
  persist(
    (set, get) => ({
      ...initialState,
      
      // 切换侧边栏
      toggleSidebar: () => {
        set((state) => ({ sidebarOpen: !state.sidebarOpen }));
      },
      
      // 切换移动端侧边栏
      toggleMobileSidebar: () => {
        set((state) => ({ mobileSidebarOpen: !state.mobileSidebarOpen }));
      },
      
      // 设置活动页面
      setActivePage: (page: string) => {
        set({ activePage: page });
      },
      
      // 切换通知面板
      toggleNotifications: () => {
        set((state) => ({ notificationsOpen: !state.notificationsOpen }));
      },
      
      // 切换搜索面板
      toggleSearch: () => {
        set((state) => ({ searchOpen: !state.searchOpen }));
      },
      
      // 切换设置面板
      toggleSettings: () => {
        set((state) => ({ settingsOpen: !state.settingsOpen }));
      },
      
      // 设置加载状态
      setLoading: (loading: boolean, text: string = '加载中...') => {
        set({ isLoading: loading, loadingText: text });
      },
      
      // 设置错误消息
      setError: (message: string | null) => {
        set({ errorMessage: message });
        
        // 自动清除错误消息
        if (message) {
          setTimeout(() => {
            set({ errorMessage: null });
          }, 5000);
        }
      },
      
      // 设置成功消息
      setSuccess: (message: string | null) => {
        set({ successMessage: message });
        
        // 自动清除成功消息
        if (message) {
          setTimeout(() => {
            set({ successMessage: null });
          }, 3000);
        }
      },
      
      // 设置警告消息
      setWarning: (message: string | null) => {
        set({ warningMessage: message });
        
        // 自动清除警告消息
        if (message) {
          setTimeout(() => {
            set({ warningMessage: null });
          }, 4000);
        }
      },
      
      // 显示确认对话框
      showConfirmDialog: (
        title: string, 
        message: string, 
        onConfirm: () => void, 
        onCancel?: () => void
      ) => {
        set({
          confirmDialog: {
            open: true,
            title,
            message,
            onConfirm,
            onCancel: onCancel || (() => {}),
          },
        });
      },
      
      // 隐藏确认对话框
      hideConfirmDialog: () => {
        set({
          confirmDialog: {
            ...get().confirmDialog,
            open: false,
          },
        });
      },
      
      // 打开模态框
      openModal: (modalId: string) => {
        set((state) => ({
          modals: {
            ...state.modals,
            [modalId]: true,
          },
        }));
      },
      
      // 关闭模态框
      closeModal: (modalId: string) => {
        set((state) => ({
          modals: {
            ...state.modals,
            [modalId]: false,
          },
        }));
      },
      
      // 切换模态框
      toggleModal: (modalId: string) => {
        set((state) => ({
          modals: {
            ...state.modals,
            [modalId]: !state.modals[modalId],
          },
        }));
      },
      
      // 重置UI状态
      resetUI: () => {
        set(initialState);
      },
    }),
    {
      name: 'ui-storage', // localStorage key
      partialize: (state) => ({
        sidebarOpen: state.sidebarOpen,
        activePage: state.activePage,
      }), // 只持久化部分状态
    }
  )
);

// UI store钩子函数（便捷方法）
export const useSidebar = () => {
  const { sidebarOpen, toggleSidebar } = useUIStore();
  return { sidebarOpen, toggleSidebar };
};

export const useMobileSidebar = () => {
  const { mobileSidebarOpen, toggleMobileSidebar } = useUIStore();
  return { mobileSidebarOpen, toggleMobileSidebar };
};

export const useLoading = () => {
  const { isLoading, loadingText, setLoading } = useUIStore();
  return { isLoading, loadingText, setLoading };
};

export const useMessages = () => {
  const { 
    errorMessage, 
    successMessage, 
    warningMessage, 
    setError, 
    setSuccess, 
    setWarning 
  } = useUIStore();
  
  return {
    errorMessage,
    successMessage,
    warningMessage,
    setError,
    setSuccess,
    setWarning,
  };
};

export const useConfirmDialog = () => {
  const { confirmDialog, showConfirmDialog, hideConfirmDialog } = useUIStore();
  return { confirmDialog, showConfirmDialog, hideConfirmDialog };
};

export const useModal = (modalId: string) => {
  const modals = useUIStore((state) => state.modals);
  const openModal = useUIStore((state) => state.openModal);
  const closeModal = useUIStore((state) => state.closeModal);
  const toggleModal = useUIStore((state) => state.toggleModal);
  
  const isOpen = modals[modalId] || false;
  
  return {
    isOpen,
    open: () => openModal(modalId),
    close: () => closeModal(modalId),
    toggle: () => toggleModal(modalId),
  };
};