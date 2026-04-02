/**
 * 主题切换组件 (Tailwind CSS版本)
 * 提供明暗模式切换功能，支持自动模式
 */

import React from 'react';
import { 
  Sun, 
  Moon, 
  Monitor, 
  Settings, 
  Download,
  Upload,
  Check,
  RotateCcw
} from 'lucide-react';
import { Menu, MenuButton, MenuItem, MenuItems } from '@headlessui/react';
import useThemeStore from '../stores/theme.store';
import { ThemeMode } from '../types/theme';

interface ThemeToggleProps {
  showSettings?: boolean;
  compact?: boolean;
  className?: string;
  position?: 'toolbar' | 'sidebar' | 'floating' | 'inline';
  size?: 'small' | 'medium' | 'large';
}

const ThemeToggle: React.FC<ThemeToggleProps> = ({
  showSettings = true,
  compact = false,
  className = '',
  position = 'toolbar',
  size = 'medium',
}) => {
  const themeStore = useThemeStore();
  const { currentTheme, config, loading } = themeStore;
  
  // 处理主题切换
  const handleToggleTheme = () => {
    const newTheme = currentTheme === 'light' ? 'dark' : 'light';
    themeStore.switchTheme(newTheme, 'user');
  };
  
  // 处理模式切换
  const handleModeChange = (mode: ThemeMode) => {
    themeStore.switchMode(mode);
  };
  
  // 处理重置
  const handleReset = () => {
    themeStore.resetToDefault();
  };
  
  // 处理导出
  const handleExport = () => {
    const data = themeStore.exportConfig();
    
    // 创建下载
    const blob = new Blob([data], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `theme-config-${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };
  
  // 处理导入
  const handleImport = () => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.json';
    input.onchange = (e: Event) => {
      const target = e.target as HTMLInputElement;
      const file = target.files?.[0];
      if (!file) return;
      
      const reader = new FileReader();
      reader.onload = (e: ProgressEvent<FileReader>) => {
        const content = e.target?.result as string;
        themeStore.importConfig(content);
      };
      reader.readAsText(file);
    };
    input.click();
  };
  
  // 渲染主切换按钮
  const renderToggleButton = () => {
    const Icon = currentTheme === 'light' ? Moon : Sun;
    const tooltip = currentTheme === 'light' ? '切换到暗色模式' : '切换到亮色模式';
    
    const sizeClasses = {
      small: 'h-8 w-8',
      medium: 'h-10 w-10',
      large: 'h-12 w-12',
    };
    
    return (
      <button
        onClick={handleToggleTheme}
        title={tooltip}
        className={`
          flex items-center justify-center rounded-lg
          bg-gray-100 dark:bg-gray-800
          text-gray-700 dark:text-gray-300
          hover:bg-gray-200 dark:hover:bg-gray-700
          transition-colors duration-200
          ${sizeClasses[size]}
          ${className}
        `}
        disabled={loading}
      >
        <Icon className="h-5 w-5" />
      </button>
    );
  };
  
  // 渲染设置菜单
  const renderSettingsMenu = () => (
    <Menu as="div" className="relative">
      <MenuButton
        className={`
          flex items-center justify-center rounded-lg
          bg-gray-100 dark:bg-gray-800
          text-gray-700 dark:text-gray-300
          hover:bg-gray-200 dark:hover:bg-gray-700
          transition-colors duration-200
          h-10 w-10
        `}
        disabled={loading}
      >
        <Settings className="h-5 w-5" />
      </MenuButton>
      
      <MenuItems className="absolute right-0 mt-2 w-56 origin-top-right rounded-md bg-white dark:bg-gray-800 shadow-lg ring-1 ring-black ring-opacity-5 focus:outline-none z-50">
        <div className="py-1">
          <MenuItem>
            {({ active }) => (
              <button
                onClick={() => handleModeChange(ThemeMode.LIGHT)}
                className={`
                  flex w-full items-center px-4 py-2 text-sm
                  ${active ? 'bg-gray-100 dark:bg-gray-700' : ''}
                  ${config.mode === ThemeMode.LIGHT ? 'text-gray-800 dark:text-gray-400' : 'text-gray-700 dark:text-gray-300'}
                `}
              >
                <Sun className="mr-3 h-5 w-5" />
                亮色模式
                {config.mode === ThemeMode.LIGHT && <Check className="ml-auto h-5 w-5" />}
              </button>
            )}
          </MenuItem>
          
          <MenuItem>
            {({ active }) => (
              <button
                onClick={() => handleModeChange(ThemeMode.DARK)}
                className={`
                  flex w-full items-center px-4 py-2 text-sm
                  ${active ? 'bg-gray-100 dark:bg-gray-700' : ''}
                  ${config.mode === ThemeMode.DARK ? 'text-gray-800 dark:text-gray-400' : 'text-gray-700 dark:text-gray-300'}
                `}
              >
                <Moon className="mr-3 h-5 w-5" />
                暗色模式
                {config.mode === ThemeMode.DARK && <Check className="ml-auto h-5 w-5" />}
              </button>
            )}
          </MenuItem>
          
          <MenuItem>
            {({ active }) => (
              <button
                onClick={() => handleModeChange(ThemeMode.AUTO)}
                className={`
                  flex w-full items-center px-4 py-2 text-sm
                  ${active ? 'bg-gray-100 dark:bg-gray-700' : ''}
                  ${config.mode === ThemeMode.AUTO ? 'text-gray-800 dark:text-gray-400' : 'text-gray-700 dark:text-gray-300'}
                `}
              >
                <Monitor className="mr-3 h-5 w-5" />
                自动模式
                {config.mode === ThemeMode.AUTO && <Check className="ml-auto h-5 w-5" />}
              </button>
            )}
          </MenuItem>
          
          <div className="border-t border-gray-200 dark:border-gray-700 my-1" />
          
          <MenuItem>
            {({ active }) => (
              <button
                onClick={handleReset}
                className={`
                  flex w-full items-center px-4 py-2 text-sm
                  ${active ? 'bg-gray-100 dark:bg-gray-700' : ''}
                  text-gray-700 dark:text-gray-300
                `}
              >
                <RotateCcw className="mr-3 h-5 w-5" />
                重置到默认
              </button>
            )}
          </MenuItem>
          
          <MenuItem>
            {({ active }) => (
              <button
                onClick={handleExport}
                className={`
                  flex w-full items-center px-4 py-2 text-sm
                  ${active ? 'bg-gray-100 dark:bg-gray-700' : ''}
                  text-gray-700 dark:text-gray-300
                `}
              >
                <Download className="mr-3 h-5 w-5" />
                导出配置
              </button>
            )}
          </MenuItem>
          
          <MenuItem>
            {({ active }) => (
              <button
                onClick={handleImport}
                className={`
                  flex w-full items-center px-4 py-2 text-sm
                  ${active ? 'bg-gray-100 dark:bg-gray-700' : ''}
                  text-gray-700 dark:text-gray-300
                `}
              >
                <Upload className="mr-3 h-5 w-5" />
                导入配置
              </button>
            )}
          </MenuItem>
        </div>
      </MenuItems>
    </Menu>
  );
  
  // 根据位置渲染不同的布局
  const renderByPosition = () => {
    switch (position) {
      case 'floating':
        return (
          <div className="fixed bottom-4 right-4 flex flex-col items-end space-y-2 z-40">
            {renderToggleButton()}
            {showSettings && renderSettingsMenu()}
          </div>
        );
      
      case 'sidebar':
        return (
          <div className="flex flex-col items-center space-y-4 p-4">
            {renderToggleButton()}
            {showSettings && renderSettingsMenu()}
          </div>
        );
      
      case 'inline':
        return (
          <div className="inline-flex items-center space-x-2">
            {renderToggleButton()}
            {showSettings && renderSettingsMenu()}
          </div>
        );
      
      case 'toolbar':
      default:
        return (
          <div className="flex items-center space-x-2">
            {renderToggleButton()}
            {showSettings && renderSettingsMenu()}
          </div>
        );
    }
  };
  
  // 紧凑模式
  if (compact) {
    return (
      <div className={className}>
        {renderToggleButton()}
      </div>
    );
  }
  
  return renderByPosition();
};

export default ThemeToggle;