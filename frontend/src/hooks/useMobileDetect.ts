/**
 * 移动端检测钩子
 * 用于检测当前设备是否为移动端，并获取屏幕尺寸信息
 */

import { useState, useEffect } from 'react';

export interface ScreenSize {
  width: number;
  height: number;
  isMobile: boolean;
  isTablet: boolean;
  isDesktop: boolean;
  orientation: 'portrait' | 'landscape';
}

export interface MobileDetectResult extends ScreenSize {
  // 设备类型检测
  isMobileDevice: boolean;
  isTabletDevice: boolean;
  isDesktopDevice: boolean;
  
  // 屏幕尺寸分类
  isSmallScreen: boolean;    // < 640px
  isMediumScreen: boolean;   // 640px - 768px
  isLargeScreen: boolean;    // 768px - 1024px
  isExtraLargeScreen: boolean; // 1024px - 1280px
  is2XLScreen: boolean;      // > 1280px
  
  // 触控支持
  isTouchDevice: boolean;
  
  // 方向变化
  isPortrait: boolean;
  isLandscape: boolean;
}

// Tailwind断点对应值
const BREAKPOINTS = {
  sm: 640,
  md: 768,
  lg: 1024,
  xl: 1280,
  '2xl': 1536,
} as const;

/**
 * 检测是否为移动端设备
 * 基于用户代理字符串和屏幕尺寸
 */
export function useMobileDetect(): MobileDetectResult {
  const [screenSize, setScreenSize] = useState<ScreenSize>(() => {
    // 初始状态使用window对象（如果可用）
    if (typeof window === 'undefined') {
      return {
        width: 0,
        height: 0,
        isMobile: false,
        isTablet: false,
        isDesktop: true,
        orientation: 'portrait',
      };
    }

    return getScreenSize();
  });

  // 用户代理检测（仅在客户端执行）
  const [userAgentInfo, setUserAgentInfo] = useState({
    isMobileDevice: false,
    isTabletDevice: false,
    isDesktopDevice: true,
    isTouchDevice: false,
  });

  useEffect(() => {
    // 仅在客户端执行
    if (typeof window === 'undefined') return;

    // 检测用户代理
    const userAgent = navigator.userAgent.toLowerCase();
    const isMobileDevice = /android|webos|iphone|ipad|ipod|blackberry|iemobile|opera mini/i.test(userAgent);
    const isTabletDevice = /ipad|android(?!.*mobile)/i.test(userAgent);
    const isDesktopDevice = !isMobileDevice && !isTabletDevice;
    
    // 检测触控支持
    const isTouchDevice = 'ontouchstart' in window || 
      (navigator.maxTouchPoints > 0) || 
      (navigator as any).msMaxTouchPoints > 0;

    setUserAgentInfo({
      isMobileDevice,
      isTabletDevice,
      isDesktopDevice,
      isTouchDevice,
    });

    // 初始屏幕尺寸
    setScreenSize(getScreenSize());

    // 窗口大小变化监听
    const handleResize = () => {
      setScreenSize(getScreenSize());
    };

    window.addEventListener('resize', handleResize);
    
    // 方向变化监听
    const handleOrientationChange = () => {
      setScreenSize(getScreenSize());
    };

    window.addEventListener('orientationchange', handleOrientationChange);

    return () => {
      window.removeEventListener('resize', handleResize);
      window.removeEventListener('orientationchange', handleOrientationChange);
    };
  }, []);

  // 计算屏幕尺寸分类
  const isSmallScreen = screenSize.width < BREAKPOINTS.sm;
  const isMediumScreen = screenSize.width >= BREAKPOINTS.sm && screenSize.width < BREAKPOINTS.md;
  const isLargeScreen = screenSize.width >= BREAKPOINTS.md && screenSize.width < BREAKPOINTS.lg;
  const isExtraLargeScreen = screenSize.width >= BREAKPOINTS.lg && screenSize.width < BREAKPOINTS.xl;
  const is2XLScreen = screenSize.width >= BREAKPOINTS.xl;

  // 基于屏幕尺寸的设备类型（覆盖用户代理检测）
  const isMobile = isSmallScreen || isMediumScreen || userAgentInfo.isMobileDevice;
  const isTablet = isLargeScreen || userAgentInfo.isTabletDevice;
  const isDesktop = isExtraLargeScreen || is2XLScreen || userAgentInfo.isDesktopDevice;

  return {
    ...screenSize,
    ...userAgentInfo,
    isMobile,
    isTablet,
    isDesktop,
    isSmallScreen,
    isMediumScreen,
    isLargeScreen,
    isExtraLargeScreen,
    is2XLScreen,
    isPortrait: screenSize.orientation === 'portrait',
    isLandscape: screenSize.orientation === 'landscape',
  };
}

/**
 * 获取当前屏幕尺寸信息
 */
function getScreenSize(): ScreenSize {
  if (typeof window === 'undefined') {
    return {
      width: 0,
      height: 0,
      isMobile: false,
      isTablet: false,
      isDesktop: true,
      orientation: 'portrait',
    };
  }

  const width = window.innerWidth;
  const height = window.innerHeight;
  const orientation = width > height ? 'landscape' : 'portrait';

  // 基于屏幕宽度的设备类型判断
  const isMobile = width < BREAKPOINTS.md;
  const isTablet = width >= BREAKPOINTS.md && width < BREAKPOINTS.lg;
  const isDesktop = width >= BREAKPOINTS.lg;

  return {
    width,
    height,
    isMobile,
    isTablet,
    isDesktop,
    orientation,
  };
}

/**
 * 简化版移动端检测钩子
 * 只返回是否为移动端
 */
export function useIsMobile(): boolean {
  const { isMobile } = useMobileDetect();
  return isMobile;
}

/**
 * 简化版平板检测钩子
 * 只返回是否为平板
 */
export function useIsTablet(): boolean {
  const { isTablet } = useMobileDetect();
  return isTablet;
}

/**
 * 简化版桌面端检测钩子
 * 只返回是否为桌面端
 */
export function useIsDesktop(): boolean {
  const { isDesktop } = useMobileDetect();
  return isDesktop;
}

/**
 * 屏幕尺寸断点检测钩子
 * 返回当前屏幕所在的断点范围
 */
export function useBreakpoint(): {
  current: keyof typeof BREAKPOINTS | null;
  isAbove: (breakpoint: keyof typeof BREAKPOINTS) => boolean;
  isBelow: (breakpoint: keyof typeof BREAKPOINTS) => boolean;
} {
  const { width } = useMobileDetect();

  const current = (() => {
    if (width >= BREAKPOINTS['2xl']) return '2xl';
    if (width >= BREAKPOINTS.xl) return 'xl';
    if (width >= BREAKPOINTS.lg) return 'lg';
    if (width >= BREAKPOINTS.md) return 'md';
    if (width >= BREAKPOINTS.sm) return 'sm';
    return null;
  })();

  const isAbove = (breakpoint: keyof typeof BREAKPOINTS) => {
    return width >= BREAKPOINTS[breakpoint];
  };

  const isBelow = (breakpoint: keyof typeof BREAKPOINTS) => {
    return width < BREAKPOINTS[breakpoint];
  };

  return { current, isAbove, isBelow };
}