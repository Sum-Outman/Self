/**
 * 移动端性能优化钩子
 * 提供移动端性能优化功能，包括：防抖、节流、请求空闲、图片懒加载、资源预加载等
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import { useMobileDetect } from './useMobileDetect';

export interface PerformanceOptions {
  /** 是否启用防抖 */
  debounce?: boolean;
  /** 防抖延迟（毫秒） */
  debounceDelay?: number;
  /** 是否启用节流 */
  throttle?: boolean;
  /** 节流延迟（毫秒） */
  throttleDelay?: number;
  /** 是否启用请求空闲调度 */
  requestIdleCallback?: boolean;
  /** 空闲回调超时（毫秒） */
  idleTimeout?: number;
  /** 是否启用图片懒加载 */
  lazyLoadImages?: boolean;
  /** 是否启用资源预加载 */
  preloadResources?: boolean;
  /** 是否启用动画优化 */
  optimizeAnimations?: boolean;
  /** 是否启用内存优化 */
  optimizeMemory?: boolean;
}

export interface PerformanceMetrics {
  /** 内存使用情况（如果可用） */
  memory?: {
    usedJSHeapSize: number;
    totalJSHeapSize: number;
    jsHeapSizeLimit: number;
  };
  /** 帧率（FPS） */
  fps: number;
  /** 渲染时间（毫秒） */
  renderTime: number;
  /** 是否内存压力 */
  isMemoryPressure: boolean;
  /** 是否CPU压力 */
  isCpuPressure: boolean;
}

export interface UseMobilePerformanceResult {
  /** 防抖函数 */
  debounce: <T extends (...args: any[]) => any>(func: T, delay?: number) => T;
  /** 节流函数 */
  throttle: <T extends (...args: any[]) => any>(func: T, delay?: number) => T;
  /** 请求空闲回调 */
  requestIdleCallback: (callback: IdleRequestCallback, options?: IdleRequestOptions) => number;
  /** 取消空闲回调 */
  cancelIdleCallback: (handle: number) => void;
  /** 图片懒加载 */
  lazyLoadImage: (element: HTMLImageElement, src: string) => void;
  /** 资源预加载 */
  preloadResource: (url: string, type: 'image' | 'script' | 'style' | 'font') => void;
  /** 清理资源 */
  cleanupResources: () => void;
  /** 性能指标 */
  metrics: PerformanceMetrics;
  /** 是否低性能设备 */
  isLowPerformanceDevice: boolean;
  /** 优化建议 */
  optimizationSuggestions: string[];
}

/**
 * 使用移动端性能优化
 * @param options 性能优化选项
 * @returns 性能优化工具和指标
 */
export function useMobilePerformance(options: PerformanceOptions = {}): UseMobilePerformanceResult {
  const { isMobileDevice, isLowPerformance } = useMobilePerformanceDetection();
  const [metrics, setMetrics] = useState<PerformanceMetrics>({
    fps: 60,
    renderTime: 0,
    isMemoryPressure: false,
    isCpuPressure: false,
  });
  
  const [optimizationSuggestions, setOptimizationSuggestions] = useState<string[]>([]);
  const animationRef = useRef<number>();
  const frameCount = useRef(0);
  const lastTime = useRef(performance.now());
  const memoryCheckInterval = useRef<NodeJS.Timeout>();
  const loadedImages = useRef<Set<string>>(new Set());
  const preloadedResources = useRef<Set<string>>(new Set());
  
  // 默认选项
  const defaultOptions: PerformanceOptions = {
    debounce: true,
    debounceDelay: 300,
    throttle: true,
    throttleDelay: 100,
    requestIdleCallback: true,
    idleTimeout: 1000,
    lazyLoadImages: true,
    preloadResources: true,
    optimizeAnimations: true,
    optimizeMemory: true,
  };
  
  const mergedOptions = { ...defaultOptions, ...options };
  
  // 防抖函数
  const debounce = useCallback(<T extends (...args: any[]) => any>(func: T, delay?: number): T => {
    if (!mergedOptions.debounce) return func;
    
    const debounceDelay = delay ?? mergedOptions.debounceDelay ?? 300;
    let timeoutId: NodeJS.Timeout;
    
    return ((...args: Parameters<T>) => {
      if (timeoutId) {
        clearTimeout(timeoutId);
      }
      
      timeoutId = setTimeout(() => {
        func(...args);
      }, debounceDelay);
    }) as T;
  }, [mergedOptions.debounce, mergedOptions.debounceDelay]);
  
  // 节流函数
  const throttle = useCallback(<T extends (...args: any[]) => any>(func: T, delay?: number): T => {
    if (!mergedOptions.throttle) return func;
    
    const throttleDelay = delay ?? mergedOptions.throttleDelay ?? 100;
    let lastExecTime = 0;
    let timeoutId: NodeJS.Timeout;
    
    return ((...args: Parameters<T>) => {
      const now = Date.now();
      const timeSinceLastExec = now - lastExecTime;
      
      if (timeSinceLastExec >= throttleDelay) {
        func(...args);
        lastExecTime = now;
      } else if (!timeoutId) {
        timeoutId = setTimeout(() => {
          func(...args);
          lastExecTime = Date.now();
          timeoutId = undefined as any;
        }, throttleDelay - timeSinceLastExec);
      }
    }) as T;
  }, [mergedOptions.throttle, mergedOptions.throttleDelay]);
  
  // 请求空闲回调（polyfill）
  const requestIdleCallback = useCallback((callback: IdleRequestCallback, options?: IdleRequestOptions): number => {
    if (!mergedOptions.requestIdleCallback) {
      return setTimeout(() => callback({ didTimeout: false, timeRemaining: () => 50 }), 0) as unknown as number;
    }
    
    if ('requestIdleCallback' in window) {
      return window.requestIdleCallback(callback, options);
    } else {
      // polyfill
      const startTime = performance.now();
      const timeout = options?.timeout ?? mergedOptions.idleTimeout ?? 1000;
      
      return setTimeout(() => {
        const elapsed = performance.now() - startTime;
        const timeRemaining = Math.max(0, 50 - elapsed);
        callback({
          didTimeout: elapsed >= timeout,
          timeRemaining: () => timeRemaining,
        });
      }, 0) as unknown as number;
    }
  }, [mergedOptions.requestIdleCallback, mergedOptions.idleTimeout]);
  
  // 取消空闲回调
  const cancelIdleCallback = useCallback((handle: number) => {
    if ('cancelIdleCallback' in window) {
      window.cancelIdleCallback(handle);
    } else {
      clearTimeout(handle);
    }
  }, []);
  
  // 图片懒加载
  const lazyLoadImage = useCallback((element: HTMLImageElement, src: string) => {
    if (!mergedOptions.lazyLoadImages) {
      element.src = src;
      return;
    }
    
    if (loadedImages.current.has(src)) {
      element.src = src;
      return;
    }
    
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            const img = entry.target as HTMLImageElement;
            img.src = src;
            loadedImages.current.add(src);
            observer.unobserve(img);
          }
        });
      },
      {
        root: null,
        rootMargin: '50px',
        threshold: 0.1,
      }
    );
    
    observer.observe(element);
  }, [mergedOptions.lazyLoadImages]);
  
  // 资源预加载
  const preloadResource = useCallback((url: string, type: 'image' | 'script' | 'style' | 'font') => {
    if (!mergedOptions.preloadResources) return;
    
    if (preloadedResources.current.has(url)) {
      return;
    }
    
    requestIdleCallback(() => {
      switch (type) {
        case 'image':
          const img = new Image();
          img.src = url;
          break;
        
        case 'script':
          const script = document.createElement('script');
          script.src = url;
          script.async = true;
          document.head.appendChild(script);
          break;
        
        case 'style':
          const link = document.createElement('link');
          link.rel = 'stylesheet';
          link.href = url;
          document.head.appendChild(link);
          break;
        
        case 'font':
          const fontLink = document.createElement('link');
          fontLink.rel = 'preload';
          fontLink.href = url;
          fontLink.as = 'font';
          fontLink.type = 'font/woff2';
          fontLink.crossOrigin = 'anonymous';
          document.head.appendChild(fontLink);
          break;
      }
      
      preloadedResources.current.add(url);
    });
  }, [mergedOptions.preloadResources, requestIdleCallback]);
  
  // 清理资源
  const cleanupResources = useCallback(() => {
    // 清理图片
    loadedImages.current.clear();
    
    // 清理预加载资源
    preloadedResources.current.clear();
    
    // 尝试触发垃圾回收（仅在支持的环境中）
    try {
      // 检查是否在浏览器环境中且gc函数可用
      if (typeof window !== 'undefined' && (window as any).gc && typeof (window as any).gc === 'function') {
        requestIdleCallback(() => {
          (window as any).gc();
        });
      }
      // 检查Node.js环境中的global.gc
      else if (typeof global !== 'undefined' && (global as any).gc && typeof (global as any).gc === 'function') {
        requestIdleCallback(() => {
          (global as any).gc();
        });
      }
    } catch (error) {
      // 忽略错误，垃圾回收不是必需功能
      console.debug('垃圾回收调用失败:', error);
    }
  }, [requestIdleCallback]);
  
  // 帧率监控
  useEffect(() => {
    if (!mergedOptions.optimizeAnimations) return;
    
    const calculateFPS = () => {
      frameCount.current++;
      const now = performance.now();
      const elapsed = now - lastTime.current;
      
      if (elapsed >= 1000) {
        const fps = Math.round((frameCount.current * 1000) / elapsed);
        
        setMetrics(prev => ({
          ...prev,
          fps,
        }));
        
        frameCount.current = 0;
        lastTime.current = now;
      }
      
      animationRef.current = requestAnimationFrame(calculateFPS);
    };
    
    animationRef.current = requestAnimationFrame(calculateFPS);
    
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [mergedOptions.optimizeAnimations]);
  
  // 内存监控
  useEffect(() => {
    if (!mergedOptions.optimizeMemory) return;
    
    const checkMemory = () => {
      if ('memory' in performance && performance.memory) {
        const memoryInfo = (performance as any).memory;
        
        setMetrics(prev => ({
          ...prev,
          memory: {
            usedJSHeapSize: memoryInfo.usedJSHeapSize,
            totalJSHeapSize: memoryInfo.totalJSHeapSize,
            jsHeapSizeLimit: memoryInfo.jsHeapSizeLimit,
          },
          isMemoryPressure: memoryInfo.usedJSHeapSize > memoryInfo.jsHeapSizeLimit * 0.8,
        }));
      }
    };
    
    // 初始检查
    checkMemory();
    
    // 定期检查
    memoryCheckInterval.current = setInterval(checkMemory, 10000);
    
    return () => {
      if (memoryCheckInterval.current) {
        clearInterval(memoryCheckInterval.current);
      }
    };
  }, [mergedOptions.optimizeMemory]);
  
  // CPU压力检测
  useEffect(() => {
    if (!mergedOptions.optimizeAnimations) return;
    
    let lastFrameTime = performance.now();
    let frameCount = 0;
    let lowFrames = 0;
    
    const checkCpuPressure = () => {
      const now = performance.now();
      const frameTime = now - lastFrameTime;
      lastFrameTime = now;
      
      frameCount++;
      
      // 如果帧时间超过16.7ms（60fps），认为是低帧率
      if (frameTime > 16.7) {
        lowFrames++;
      }
      
      // 每100帧检查一次
      if (frameCount >= 100) {
        const lowFrameRate = lowFrames / frameCount;
        
        setMetrics(prev => ({
          ...prev,
          isCpuPressure: lowFrameRate > 0.3, // 30%以上低帧率
        }));
        
        frameCount = 0;
        lowFrames = 0;
      }
      
      requestAnimationFrame(checkCpuPressure);
    };
    
    const animationId = requestAnimationFrame(checkCpuPressure);
    
    return () => {
      cancelAnimationFrame(animationId);
    };
  }, [mergedOptions.optimizeAnimations]);
  
  // 生成优化建议
  useEffect(() => {
    const suggestions: string[] = [];
    
    if (isMobileDevice) {
      suggestions.push('移动端设备：建议使用响应式设计和触摸优化');
    }
    
    if (isLowPerformance) {
      suggestions.push('低性能设备：建议减少动画效果和图片分辨率');
    }
    
    if (metrics.fps < 30) {
      suggestions.push(`低帧率（${metrics.fps}FPS）：建议减少复杂动画和DOM操作`);
    }
    
    if (metrics.isMemoryPressure) {
      suggestions.push('内存压力：建议清理未使用的资源，减少内存占用');
    }
    
    if (metrics.isCpuPressure) {
      suggestions.push('CPU压力：建议减少JavaScript计算，优化算法效率');
    }
    
    setOptimizationSuggestions(suggestions);
  }, [isMobileDevice, isLowPerformance, metrics]);
  
  return {
    debounce,
    throttle,
    requestIdleCallback,
    cancelIdleCallback,
    lazyLoadImage,
    preloadResource,
    cleanupResources,
    metrics,
    isLowPerformanceDevice: isLowPerformance,
    optimizationSuggestions,
  };
}

/**
 * 检测是否为低性能设备
 */
function useMobilePerformanceDetection() {
  const { isMobileDevice, isSmallScreen } = useMobileDetect();
  const [isLowPerformance, setIsLowPerformance] = useState(false);
  
  useEffect(() => {
    if (typeof window === 'undefined') return;
    
    // 检测设备性能
    const checkPerformance = () => {
      // 基于多个因素判断是否为低性能设备
      let score = 0;
      
      // 移动设备
      if (isMobileDevice) score += 2;
      
      // 小屏幕
      if (isSmallScreen) score += 1;
      
      // 内存限制
      if ('deviceMemory' in navigator) {
        const memory = (navigator as any).deviceMemory;
        if (memory <= 2) score += 3; // 2GB或更少
        else if (memory <= 4) score += 2; // 4GB
      }
      
      // 硬件并发性
      if ('hardwareConcurrency' in navigator) {
        const cores = navigator.hardwareConcurrency;
        if (cores <= 2) score += 2; // 2核心或更少
        else if (cores <= 4) score += 1; // 4核心
      }
      
      // 是否为低性能设备
      setIsLowPerformance(score >= 4);
    };
    
    checkPerformance();
  }, [isMobileDevice, isSmallScreen]);
  
  return { isMobileDevice, isLowPerformance };
}

/**
 * 简化版：是否低性能设备
 */
export function useIsLowPerformance(): boolean {
  const { isLowPerformance } = useMobilePerformanceDetection();
  return isLowPerformance;
}

export default useMobilePerformance;