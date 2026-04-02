/**
 * 可访问性工具
 * 提供改善可访问性的辅助函数和工具
 */

/**
 * 键盘导航工具
 */
export class KeyboardNavigation {
  private static instance: KeyboardNavigation;
  private focusableSelectors = [
    'a[href]',
    'button',
    'input',
    'select',
    'textarea',
    '[tabindex]:not([tabindex="-1"])',
    '[contenteditable="true"]',
    '[role="button"]',
    '[role="link"]',
    '[role="checkbox"]',
    '[role="radio"]',
    '[role="tab"]',
  ];
  
  private constructor() {}
  
  static getInstance(): KeyboardNavigation {
    if (!KeyboardNavigation.instance) {
      KeyboardNavigation.instance = new KeyboardNavigation();
    }
    return KeyboardNavigation.instance;
  }
  
  /**
   * 获取所有可聚焦元素
   */
  getAllFocusableElements(container: HTMLElement = document.body): HTMLElement[] {
    return Array.from(container.querySelectorAll(this.focusableSelectors.join(','))) as HTMLElement[];
  }
  
  /**
   * 限制焦点在指定容器内
   */
  trapFocus(container: HTMLElement): () => void {
    const focusableElements = this.getAllFocusableElements(container);
    if (focusableElements.length === 0) return () => {};
    
    const firstFocusableElement = focusableElements[0];
    const lastFocusableElement = focusableElements[focusableElements.length - 1];
    
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key !== 'Tab') return;
      
      if (event.shiftKey) {
        // Shift + Tab
        if (document.activeElement === firstFocusableElement) {
          event.preventDefault();
          lastFocusableElement.focus();
        }
      } else {
        // Tab
        if (document.activeElement === lastFocusableElement) {
          event.preventDefault();
          firstFocusableElement.focus();
        }
      }
    };
    
    // 初始焦点
    firstFocusableElement.focus();
    
    // 添加事件监听器
    container.addEventListener('keydown', handleKeyDown);
    
    // 返回清理函数
    return () => {
      container.removeEventListener('keydown', handleKeyDown);
    };
  }
  
  /**
   * 添加键盘快捷键
   */
  addShortcut(
    key: string,
    callback: (event: KeyboardEvent) => void,
    options: {
      ctrl?: boolean;
      shift?: boolean;
      alt?: boolean;
      meta?: boolean;
      preventDefault?: boolean;
      stopPropagation?: boolean;
    } = {}
  ): () => void {
    const {
      ctrl = false,
      shift = false,
      alt = false,
      meta = false,
      preventDefault = true,
      stopPropagation = false,
    } = options;
    
    const handleKeyDown = (event: KeyboardEvent) => {
      // 安全检查：确保event.key存在
      if (!event.key) return;
      
      if (
        event.key.toLowerCase() === key.toLowerCase() &&
        event.ctrlKey === ctrl &&
        event.shiftKey === shift &&
        event.altKey === alt &&
        event.metaKey === meta
      ) {
        if (preventDefault) event.preventDefault();
        if (stopPropagation) event.stopPropagation();
        callback(event);
      }
    };
    
    document.addEventListener('keydown', handleKeyDown);
    
    // 返回清理函数
    return () => {
      document.removeEventListener('keydown', handleKeyDown);
    };
  }
  
  /**
   * 改进可访问性的点击处理
   */
  enhanceClickAccessibility(element: HTMLElement): () => void {
    const handleKeyDown = (event: KeyboardEvent) => {
      // 安全检查：确保event.key存在
      if (!event.key) return;
      
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        element.click();
      }
    };
    
    element.setAttribute('tabindex', '0');
    element.setAttribute('role', 'button');
    
    element.addEventListener('keydown', handleKeyDown);
    
    // 返回清理函数
    return () => {
      element.removeEventListener('keydown', handleKeyDown);
    };
  }
}

/**
 * 屏幕阅读器工具
 */
export class ScreenReader {
  private static instance: ScreenReader;
  private liveRegion: HTMLElement | null = null;
  
  private constructor() {}
  
  static getInstance(): ScreenReader {
    if (!ScreenReader.instance) {
      ScreenReader.instance = new ScreenReader();
    }
    return ScreenReader.instance;
  }
  
  /**
   * 初始化实时区域
   */
  private ensureLiveRegion(): HTMLElement {
    if (!this.liveRegion) {
      this.liveRegion = document.createElement('div');
      this.liveRegion.setAttribute('aria-live', 'polite');
      this.liveRegion.setAttribute('aria-atomic', 'true');
      this.liveRegion.setAttribute('class', 'sr-only');
      this.liveRegion.style.position = 'absolute';
      this.liveRegion.style.width = '1px';
      this.liveRegion.style.height = '1px';
      this.liveRegion.style.padding = '0';
      this.liveRegion.style.margin = '-1px';
      this.liveRegion.style.overflow = 'hidden';
      this.liveRegion.style.clip = 'rect(0, 0, 0, 0)';
      this.liveRegion.style.whiteSpace = 'nowrap';
      this.liveRegion.style.border = '0';
      document.body.appendChild(this.liveRegion);
    }
    return this.liveRegion;
  }
  
  /**
   * 向屏幕阅读器朗读消息
   */
  speak(message: string, priority: 'polite' | 'assertive' = 'polite'): void {
    const liveRegion = this.ensureLiveRegion();
    liveRegion.setAttribute('aria-live', priority);
    
    // 清除现有内容
    liveRegion.textContent = '';
    
    // 添加新消息
    setTimeout(() => {
      liveRegion.textContent = message;
    }, 100);
    
    // 清理消息
    setTimeout(() => {
      liveRegion.textContent = '';
    }, 5000);
  }
  
  /**
   * 通知操作结果
   */
  notify(
    message: string,
    type: 'success' | 'error' | 'warning' | 'info' = 'info'
  ): void {
    const priority = type === 'error' ? 'assertive' : 'polite';
    const prefix = {
      success: '成功：',
      error: '错误：',
      warning: '警告：',
      info: '信息：',
    }[type];
    
    this.speak(`${prefix}${message}`, priority);
  }
}

/**
 * 高对比度模式工具
 */
export class HighContrastMode {
  private static instance: HighContrastMode;
  private isEnabled = false;
  private styleElement: HTMLStyleElement | null = null;
  
  private constructor() {
    this.detect();
  }
  
  static getInstance(): HighContrastMode {
    if (!HighContrastMode.instance) {
      HighContrastMode.instance = new HighContrastMode();
    }
    return HighContrastMode.instance;
  }
  
  /**
   * 检测系统高对比度设置
   */
  private detect(): void {
    // 检测Windows高对比度模式
    if (window.matchMedia) {
      const highContrastQuery = window.matchMedia('(forced-colors: active)');
      this.isEnabled = highContrastQuery.matches;
      
      highContrastQuery.addEventListener('change', (event) => {
        this.isEnabled = event.matches;
        this.updateStyles();
      });
    }
    
    // 检测用户偏好
    const userPrefersContrast = window.matchMedia('(prefers-contrast: more)').matches;
    if (userPrefersContrast) {
      this.isEnabled = true;
    }
  }
  
  /**
   * 更新样式
   */
  private updateStyles(): void {
    if (this.isEnabled) {
      this.enable();
    } else {
      this.disable();
    }
  }
  
  /**
   * 启用高对比度样式
   */
  enable(): void {
    if (!this.styleElement) {
      this.styleElement = document.createElement('style');
      this.styleElement.textContent = this.getHighContrastStyles();
      document.head.appendChild(this.styleElement);
    }
  }
  
  /**
   * 禁用高对比度样式
   */
  disable(): void {
    if (this.styleElement) {
      document.head.removeChild(this.styleElement);
      this.styleElement = null;
    }
  }
  
  /**
   * 获取高对比度样式
   */
  private getHighContrastStyles(): string {
    return `
      /* 高对比度模式样式 */
      body {
        --primary-color: #0000FF;
        --primary-contrast: #FFFFFF;
        --secondary-color: #FF0000;
        --secondary-contrast: #FFFFFF;
        --accent-color: #008000;
        --accent-contrast: #FFFFFF;
        --error-color: #FF0000;
        --warning-color: #FFA500;
        --success-color: #008000;
        --info-color: #0000FF;
        --text-color: #000000;
        --text-contrast: #FFFFFF;
        --background-color: #FFFFFF;
        --surface-color: #FFFFFF;
        --border-color: #000000;
      }
      
      * {
        color: var(--text-color) !important;
        background-color: var(--background-color) !important;
        border-color: var(--border-color) !important;
        text-decoration: underline !important;
      }
      
      button, a, input, select, textarea {
        border: 2px solid var(--border-color) !important;
      }
      
      .focus-visible {
        outline: 3px solid var(--primary-color) !important;
        outline-offset: 2px !important;
      }
    `;
  }
  
  /**
   * 检查是否启用高对比度模式
   */
  isHighContrastEnabled(): boolean {
    return this.isEnabled;
  }
  
  /**
   * 切换高对比度模式
   */
  toggle(): void {
    this.isEnabled = !this.isEnabled;
    this.updateStyles();
  }
}

/**
 * 减少动画工具
 */
export class ReducedMotion {
  private static instance: ReducedMotion;
  private isEnabled = false;
  
  private constructor() {
    this.detect();
  }
  
  static getInstance(): ReducedMotion {
    if (!ReducedMotion.instance) {
      ReducedMotion.instance = new ReducedMotion();
    }
    return ReducedMotion.instance;
  }
  
  /**
   * 检测减少动画偏好
   */
  private detect(): void {
    if (window.matchMedia) {
      const reducedMotionQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
      this.isEnabled = reducedMotionQuery.matches;
      
      reducedMotionQuery.addEventListener('change', (event) => {
        this.isEnabled = event.matches;
      });
    }
  }
  
  /**
   * 检查是否启用减少动画
   */
  isReducedMotionEnabled(): boolean {
    return this.isEnabled;
  }
  
  /**
   * 应用减少动画
   */
  applyReducedMotion(element: HTMLElement): void {
    if (this.isEnabled) {
      element.style.animationDuration = '0.001ms';
      element.style.transitionDuration = '0.001ms';
      element.style.scrollBehavior = 'auto';
    }
  }
  
  /**
   * 移除减少动画
   */
  removeReducedMotion(element: HTMLElement): void {
    element.style.animationDuration = '';
    element.style.transitionDuration = '';
    element.style.scrollBehavior = '';
  }
}

/**
 * 字体大小工具
 */
export class FontSizeManager {
  private static instance: FontSizeManager;
  private baseFontSize = 16;
  private currentMultiplier = 1.0;
  private readonly minMultiplier = 0.5;
  private readonly maxMultiplier = 3.0;
  
  private constructor() {
    this.loadFromStorage();
  }
  
  static getInstance(): FontSizeManager {
    if (!FontSizeManager.instance) {
      FontSizeManager.instance = new FontSizeManager();
    }
    return FontSizeManager.instance;
  }
  
  /**
   * 从存储中加载设置
   */
  private loadFromStorage(): void {
    try {
      const saved = localStorage.getItem('fontSizeMultiplier');
      if (saved) {
        this.currentMultiplier = parseFloat(saved);
      }
    } catch (error) {
      console.error('加载字体大小设置失败:', error);
    }
  }
  
  /**
   * 保存到存储
   */
  private saveToStorage(): void {
    try {
      localStorage.setItem('fontSizeMultiplier', this.currentMultiplier.toString());
    } catch (error) {
      console.error('保存字体大小设置失败:', error);
    }
  }
  
  /**
   * 增大字体
   */
  increase(): void {
    const newMultiplier = Math.min(
      this.currentMultiplier + 0.1,
      this.maxMultiplier
    );
    this.setMultiplier(newMultiplier);
  }
  
  /**
   * 减小字体
   */
  decrease(): void {
    const newMultiplier = Math.max(
      this.currentMultiplier - 0.1,
      this.minMultiplier
    );
    this.setMultiplier(newMultiplier);
  }
  
  /**
   * 重置字体大小
   */
  reset(): void {
    this.setMultiplier(1.0);
  }
  
  /**
   * 设置字体大小倍数
   */
  setMultiplier(multiplier: number): void {
    const clamped = Math.max(this.minMultiplier, Math.min(multiplier, this.maxMultiplier));
    this.currentMultiplier = clamped;
    
    // 更新根元素字体大小
    const root = document.documentElement;
    const computedFontSize = this.baseFontSize * clamped;
    root.style.fontSize = `${computedFontSize}px`;
    
    // 保存设置
    this.saveToStorage();
  }
  
  /**
   * 获取当前字体大小倍数
   */
  getMultiplier(): number {
    return this.currentMultiplier;
  }
  
  /**
   * 获取字体大小级别
   */
  getSizeLevel(): 'small' | 'medium' | 'large' | 'xlarge' {
    if (this.currentMultiplier < 0.8) return 'small';
    if (this.currentMultiplier < 1.2) return 'medium';
    if (this.currentMultiplier < 1.8) return 'large';
    return 'xlarge';
  }
  
  /**
   * 设置字体大小级别
   */
  setSizeLevel(level: 'small' | 'medium' | 'large' | 'xlarge'): void {
    const multipliers = {
      small: 0.75,
      medium: 1.0,
      large: 1.5,
      xlarge: 2.0,
    };
    this.setMultiplier(multipliers[level]);
  }
}

/**
 * 颜色对比度计算工具
 */
export class ColorContrast {
  /**
   * 计算亮度
   */
  private static calculateLuminance(r: number, g: number, b: number): number {
    const sRGB = [r, g, b].map((c) => {
      const normalized = c / 255;
      return normalized <= 0.03928
        ? normalized / 12.92
        : Math.pow((normalized + 0.055) / 1.055, 2.4);
    });
    
    return 0.2126 * sRGB[0] + 0.7152 * sRGB[1] + 0.0722 * sRGB[2];
  }
  
  /**
   * 解析颜色
   */
  private static parseColor(color: string): [number, number, number] | null {
    // 移除空格和转换为小写
    color = color.trim().toLowerCase();
    
    // HEX格式
    if (color.startsWith('#')) {
      const hex = color.slice(1);
      if (hex.length === 3) {
        const r = parseInt(hex[0] + hex[0], 16);
        const g = parseInt(hex[1] + hex[1], 16);
        const b = parseInt(hex[2] + hex[2], 16);
        return [r, g, b];
      } else if (hex.length === 6) {
        const r = parseInt(hex.slice(0, 2), 16);
        const g = parseInt(hex.slice(2, 4), 16);
        const b = parseInt(hex.slice(4, 6), 16);
        return [r, g, b];
      }
    }
    
    // RGB格式
    const rgbMatch = color.match(/rgb\((\d+),\s*(\d+),\s*(\d+)\)/);
    if (rgbMatch) {
      return [parseInt(rgbMatch[1]), parseInt(rgbMatch[2]), parseInt(rgbMatch[3])];
    }
    
    // RGBA格式
    const rgbaMatch = color.match(/rgba\((\d+),\s*(\d+),\s*(\d+),\s*[\d.]+\)/);
    if (rgbaMatch) {
      return [parseInt(rgbaMatch[1]), parseInt(rgbaMatch[2]), parseInt(rgbaMatch[3])];
    }
    
    return null;
  }
  
  /**
   * 计算颜色对比度
   */
  static calculateContrast(color1: string, color2: string): number | null {
    const rgb1 = this.parseColor(color1);
    const rgb2 = this.parseColor(color2);
    
    if (!rgb1 || !rgb2) return null;
    
    const luminance1 = this.calculateLuminance(rgb1[0], rgb1[1], rgb1[2]);
    const luminance2 = this.calculateLuminance(rgb2[0], rgb2[1], rgb2[2]);
    
    const lighter = Math.max(luminance1, luminance2);
    const darker = Math.min(luminance1, luminance2);
    
    return (lighter + 0.05) / (darker + 0.05);
  }
  
  /**
   * 检查对比度是否满足WCAG标准
   */
  static checkWCAGCompliance(
    foreground: string,
    background: string
  ): {
    aa: boolean;
    aaLarge: boolean;
    aaa: boolean;
    aaaLarge: boolean;
    contrast: number;
  } {
    const contrast = this.calculateContrast(foreground, background);
    if (!contrast) {
      return {
        aa: false,
        aaLarge: false,
        aaa: false,
        aaaLarge: false,
        contrast: 0,
      };
    }
    
    return {
      aa: contrast >= 4.5,
      aaLarge: contrast >= 3.0,
      aaa: contrast >= 7.0,
      aaaLarge: contrast >= 4.5,
      contrast,
    };
  }
  
  /**
   * 生成高对比度的颜色
   */
  static generateContrastColor(baseColor: string, targetContrast: number = 4.5): string | null {
    const rgb = this.parseColor(baseColor);
    if (!rgb) return null;
    
    // 计算基础亮度
    const baseLuminance = this.calculateLuminance(rgb[0], rgb[1], rgb[2]);
    
    // 确定目标亮度（更深或更浅）
    const targetLuminance = baseLuminance > 0.5
      ? (baseLuminance + 0.05) / targetContrast - 0.05  // 更深
      : (baseLuminance + 0.05) * targetContrast - 0.05; // 更浅
    
    // 生成目标颜色
    const targetRgb = rgb.map((c) => {
      const normalized = c / 255;
      const luminanceAdjusted = targetLuminance / (normalized === 0 ? 1 : normalized);
      return Math.max(0, Math.min(255, Math.round(luminanceAdjusted * 255)));
    });
    
    return `rgb(${targetRgb[0]}, ${targetRgb[1]}, ${targetRgb[2]})`;
  }
}

// 导出单例实例
export const keyboardNavigation = KeyboardNavigation.getInstance();
export const screenReader = ScreenReader.getInstance();
export const highContrastMode = HighContrastMode.getInstance();
export const reducedMotion = ReducedMotion.getInstance();
export const fontSizeManager = FontSizeManager.getInstance();