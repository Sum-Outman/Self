/**
 * 触摸手势处理器组件
 * 提供移动端触摸手势支持：轻扫、长按、双指缩放、旋转等
 */

import React, { useRef, useEffect, useCallback, useState } from 'react';
import { cn } from '../../utils/cn';

export type GestureType = 
  | 'tap'           // 轻点
  | 'doubleTap'     // 双击
  | 'longPress'     // 长按
  | 'swipeLeft'     // 向左轻扫
  | 'swipeRight'    // 向右轻扫
  | 'swipeUp'       // 向上轻扫
  | 'swipeDown'     // 向下轻扫
  | 'pinchIn'       // 双指捏合
  | 'pinchOut'      // 双指张开
  | 'rotate'        // 旋转
  | 'pan'           // 平移
  | 'flick'         // 快速轻扫
  | 'hold'          // 按住
  | 'release';      // 释放

export interface GestureEvent {
  type: GestureType;
  position?: {
    x: number;
    y: number;
  };
  delta?: {
    x: number;
    y: number;
  };
  scale?: number;
  rotation?: number;
  velocity?: {
    x: number;
    y: number;
  };
  timestamp: number;
  target?: HTMLElement;
}

export interface TouchGestureHandlerProps extends Omit<React.HTMLAttributes<HTMLDivElement>, 'onGesture'> {
  /** 手势配置 */
  config?: {
    /** 点击阈值（毫秒） */
    tapThreshold?: number;
    /** 双击阈值（毫秒） */
    doubleTapThreshold?: number;
    /** 长按阈值（毫秒） */
    longPressThreshold?: number;
    /** 轻扫阈值（像素） */
    swipeThreshold?: number;
    /** 轻扫速度阈值（像素/毫秒） */
    swipeVelocityThreshold?: number;
    /** 缩放灵敏度 */
    pinchSensitivity?: number;
    /** 旋转灵敏度 */
    rotateSensitivity?: number;
    /** 是否启用多点触控 */
    multiTouch?: boolean;
    /** 是否阻止默认触摸行为 */
    preventDefault?: boolean;
    /** 是否启用被动事件监听器 */
    passive?: boolean;
  };
  
  /** 手势事件处理函数 */
  onGesture?: (event: GestureEvent) => void;
  
  /** 特定手势事件处理函数 */
  onTap?: (position: { x: number; y: number }) => void;
  onDoubleTap?: (position: { x: number; y: number }) => void;
  onLongPress?: (position: { x: number; y: number }) => void;
  onSwipeLeft?: () => void;
  onSwipeRight?: () => void;
  onSwipeUp?: () => void;
  onSwipeDown?: () => void;
  onPinchIn?: (scale: number) => void;
  onPinchOut?: (scale: number) => void;
  onRotate?: (rotation: number) => void;
  onPan?: (delta: { x: number; y: number }) => void;
  onFlick?: (direction: 'left' | 'right' | 'up' | 'down', velocity: number) => void;
  onHold?: () => void;
  onRelease?: () => void;
  
  /** 子元素 */
  children?: React.ReactNode;
  
  /** 是否禁用 */
  disabled?: boolean;
  
  /** 自定义类名 */
  className?: string;
}

const defaultConfig = {
  tapThreshold: 300,           // 点击阈值（毫秒）
  doubleTapThreshold: 300,     // 双击阈值（毫秒）
  longPressThreshold: 500,     // 长按阈值（毫秒）
  swipeThreshold: 30,          // 轻扫阈值（像素）
  swipeVelocityThreshold: 0.3, // 轻扫速度阈值（像素/毫秒）
  pinchSensitivity: 1.0,       // 缩放灵敏度
  rotateSensitivity: 1.0,      // 旋转灵敏度
  multiTouch: true,            // 是否启用多点触控
  preventDefault: true,        // 是否阻止默认触摸行为
  passive: false,              // 是否启用被动事件监听器
};

export const TouchGestureHandler: React.FC<TouchGestureHandlerProps> = ({
  config = {},
  onGesture,
  onTap,
  onDoubleTap,
  onLongPress,
  onSwipeLeft,
  onSwipeRight,
  onSwipeUp,
  onSwipeDown,
  onPinchIn,
  onPinchOut,
  onRotate,
  onPan,
  onFlick,
  onHold,
  onRelease,
  children,
  disabled = false,
  className,
  ...props
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [isTouching, setIsTouching] = useState(false);
  
  // 合并配置
  const mergedConfig = { ...defaultConfig, ...config };
  
  // 触摸状态
  const touchState = useRef({
    startTime: 0,
    startX: 0,
    startY: 0,
    lastX: 0,
    lastY: 0,
    lastTime: 0,
    touchCount: 0,
    isLongPress: false,
    longPressTimer: null as NodeJS.Timeout | null,
    tapCount: 0,
    lastTapTime: 0,
    isSwiping: false,
    isPinching: false,
    isRotating: false,
    initialDistance: 0,
    initialAngle: 0,
  });
  
  // 触发手势事件
  const triggerGesture = useCallback((event: GestureEvent) => {
    if (disabled) return;
    
    // 调用通用手势处理函数
    if (onGesture) {
      onGesture(event);
    }
    
    // 调用特定手势处理函数
    switch (event.type) {
      case 'tap':
        if (onTap && event.position) {
          onTap(event.position);
        }
        break;
      
      case 'doubleTap':
        if (onDoubleTap && event.position) {
          onDoubleTap(event.position);
        }
        break;
      
      case 'longPress':
        if (onLongPress && event.position) {
          onLongPress(event.position);
        }
        break;
      
      case 'swipeLeft':
        if (onSwipeLeft) onSwipeLeft();
        break;
      
      case 'swipeRight':
        if (onSwipeRight) onSwipeRight();
        break;
      
      case 'swipeUp':
        if (onSwipeUp) onSwipeUp();
        break;
      
      case 'swipeDown':
        if (onSwipeDown) onSwipeDown();
        break;
      
      case 'pinchIn':
        if (onPinchIn && event.scale) {
          onPinchIn(event.scale);
        }
        break;
      
      case 'pinchOut':
        if (onPinchOut && event.scale) {
          onPinchOut(event.scale);
        }
        break;
      
      case 'rotate':
        if (onRotate && event.rotation) {
          onRotate(event.rotation);
        }
        break;
      
      case 'pan':
        if (onPan && event.delta) {
          onPan(event.delta);
        }
        break;
      
      case 'flick':
        if (onFlick && event.velocity && event.delta) {
          const direction = 
            Math.abs(event.delta.x) > Math.abs(event.delta.y)
              ? (event.delta.x > 0 ? 'right' : 'left')
              : (event.delta.y > 0 ? 'down' : 'up');
          const velocity = Math.sqrt(event.velocity.x ** 2 + event.velocity.y ** 2);
          onFlick(direction, velocity);
        }
        break;
      
      case 'hold':
        if (onHold) onHold();
        break;
      
      case 'release':
        if (onRelease) onRelease();
        break;
    }
  }, [
    disabled, onGesture, onTap, onDoubleTap, onLongPress,
    onSwipeLeft, onSwipeRight, onSwipeUp, onSwipeDown,
    onPinchIn, onPinchOut, onRotate, onPan, onFlick,
    onHold, onRelease,
  ]);
  
  // 计算两点距离
  const getDistance = (x1: number, y1: number, x2: number, y2: number): number => {
    return Math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2);
  };
  
  // 计算两点角度
  const getAngle = (x1: number, y1: number, x2: number, y2: number): number => {
    return Math.atan2(y2 - y1, x2 - x1) * (180 / Math.PI);
  };
  
  // 触摸开始处理
  const handleTouchStart = useCallback((e: TouchEvent) => {
    if (disabled || !containerRef.current) return;
    
    if (mergedConfig.preventDefault) {
      e.preventDefault();
    }
    
    const touch = e.touches[0];
    const rect = containerRef.current.getBoundingClientRect();
    const x = touch.clientX - rect.left;
    const y = touch.clientY - rect.top;
    const now = Date.now();
    
    touchState.current = {
      ...touchState.current,
      startTime: now,
      startX: x,
      startY: y,
      lastX: x,
      lastY: y,
      lastTime: now,
      touchCount: e.touches.length,
      isLongPress: false,
      isSwiping: false,
      isPinching: false,
      isRotating: false,
    };
    
    setIsTouching(true);
    
    // 长按计时器
    if (touchState.current.longPressTimer) {
      clearTimeout(touchState.current.longPressTimer);
    }
    
    touchState.current.longPressTimer = setTimeout(() => {
      if (touchState.current.touchCount === 1 && !touchState.current.isSwiping) {
        touchState.current.isLongPress = true;
        triggerGesture({
          type: 'longPress',
          position: { x, y },
          timestamp: Date.now(),
        });
      }
    }, mergedConfig.longPressThreshold);
    
    // 多点触控检测
    if (e.touches.length === 2 && mergedConfig.multiTouch) {
      const touch1 = e.touches[0];
      const touch2 = e.touches[1];
      
      const x1 = touch1.clientX - rect.left;
      const y1 = touch1.clientY - rect.top;
      const x2 = touch2.clientX - rect.left;
      const y2 = touch2.clientY - rect.top;
      
      touchState.current.initialDistance = getDistance(x1, y1, x2, y2);
      touchState.current.initialAngle = getAngle(x1, y1, x2, y2);
      touchState.current.isPinching = true;
      touchState.current.isRotating = true;
    }
    
    // 触发hold事件
    triggerGesture({
      type: 'hold',
      timestamp: now,
    });
  }, [disabled, mergedConfig, triggerGesture]);
  
  // 触摸移动处理
  const handleTouchMove = useCallback((e: TouchEvent) => {
    if (disabled || !containerRef.current || !isTouching) return;
    
    if (mergedConfig.preventDefault) {
      e.preventDefault();
    }
    
    const touch = e.touches[0];
    const rect = containerRef.current.getBoundingClientRect();
    const x = touch.clientX - rect.left;
    const y = touch.clientY - rect.top;
    const now = Date.now();
    const deltaTime = now - touchState.current.lastTime;
    
    // 计算移动距离和速度
    const deltaX = x - touchState.current.lastX;
    const deltaY = y - touchState.current.lastY;
    const velocityX = deltaX / (deltaTime || 1);
    const velocityY = deltaY / (deltaTime || 1);
    
    // 更新状态
    touchState.current.lastX = x;
    touchState.current.lastY = y;
    touchState.current.lastTime = now;
    
    // 检测是否开始轻扫
    const totalDistance = getDistance(
      touchState.current.startX,
      touchState.current.startY,
      x,
      y
    );
    
    if (totalDistance > mergedConfig.swipeThreshold && !touchState.current.isSwiping) {
      touchState.current.isSwiping = true;
      // 取消长按计时器
      if (touchState.current.longPressTimer) {
        clearTimeout(touchState.current.longPressTimer);
        touchState.current.longPressTimer = null;
      }
    }
    
    // 处理平移
    if (touchState.current.isSwiping) {
      triggerGesture({
        type: 'pan',
        position: { x, y },
        delta: { x: deltaX, y: deltaY },
        velocity: { x: velocityX, y: velocityY },
        timestamp: now,
      });
    }
    
    // 处理多点触控（缩放和旋转）
    if (e.touches.length === 2 && mergedConfig.multiTouch) {
      const touch1 = e.touches[0];
      const touch2 = e.touches[1];
      
      const x1 = touch1.clientX - rect.left;
      const y1 = touch1.clientY - rect.top;
      const x2 = touch2.clientX - rect.left;
      const y2 = touch2.clientY - rect.top;
      
      const currentDistance = getDistance(x1, y1, x2, y2);
      const currentAngle = getAngle(x1, y1, x2, y2);
      
      // 缩放
      if (touchState.current.isPinching && touchState.current.initialDistance > 0) {
        const scale = currentDistance / touchState.current.initialDistance;
        triggerGesture({
          type: scale < 1 ? 'pinchIn' : 'pinchOut',
          scale,
          timestamp: now,
        });
      }
      
      // 旋转
      if (touchState.current.isRotating) {
        const rotation = currentAngle - touchState.current.initialAngle;
        triggerGesture({
          type: 'rotate',
          rotation,
          timestamp: now,
        });
      }
    }
  }, [disabled, mergedConfig, triggerGesture, isTouching]);
  
  // 触摸结束处理
  const handleTouchEnd = useCallback((e: TouchEvent) => {
    if (disabled || !containerRef.current || !isTouching) return;
    
    if (mergedConfig.preventDefault) {
      e.preventDefault();
    }
    
    const rect = containerRef.current.getBoundingClientRect();
    const now = Date.now();
    const elapsedTime = now - touchState.current.startTime;
    
    // 计算最终位置和距离
    let endX = touchState.current.lastX;
    let endY = touchState.current.lastY;
    
    if (e.changedTouches.length > 0) {
      const touch = e.changedTouches[0];
      endX = touch.clientX - rect.left;
      endY = touch.clientY - rect.top;
    }
    
    const deltaX = endX - touchState.current.startX;
    const deltaY = endY - touchState.current.startY;
    const totalDistance = getDistance(
      touchState.current.startX,
      touchState.current.startY,
      endX,
      endY
    );
    
    // 清除长按计时器
    if (touchState.current.longPressTimer) {
      clearTimeout(touchState.current.longPressTimer);
      touchState.current.longPressTimer = null;
    }
    
    // 检测手势类型
    if (touchState.current.isLongPress) {
      // 长按已处理
    } else if (touchState.current.isSwiping) {
      // 检测轻扫方向
      const velocity = totalDistance / elapsedTime;
      
      if (velocity > mergedConfig.swipeVelocityThreshold) {
        // 快速轻扫
        const velocityX = deltaX / elapsedTime;
        const velocityY = deltaY / elapsedTime;
        
        triggerGesture({
          type: 'flick',
          delta: { x: deltaX, y: deltaY },
          velocity: { x: velocityX, y: velocityY },
          timestamp: now,
        });
      } else {
        // 普通轻扫
        let swipeType: GestureType | null = null;
        
        if (Math.abs(deltaX) > Math.abs(deltaY)) {
          // 水平轻扫
          if (deltaX > mergedConfig.swipeThreshold) {
            swipeType = 'swipeRight';
          } else if (deltaX < -mergedConfig.swipeThreshold) {
            swipeType = 'swipeLeft';
          }
        } else {
          // 垂直轻扫
          if (deltaY > mergedConfig.swipeThreshold) {
            swipeType = 'swipeDown';
          } else if (deltaY < -mergedConfig.swipeThreshold) {
            swipeType = 'swipeUp';
          }
        }
        
        if (swipeType) {
          triggerGesture({
            type: swipeType,
            position: { x: endX, y: endY },
            timestamp: now,
          });
        }
      }
    } else if (elapsedTime < mergedConfig.tapThreshold && totalDistance < mergedConfig.swipeThreshold) {
      // 点击检测
      const tapTime = now;
      const timeSinceLastTap = tapTime - touchState.current.lastTapTime;
      
      if (timeSinceLastTap < mergedConfig.doubleTapThreshold && touchState.current.tapCount === 1) {
        // 双击
        touchState.current.tapCount = 0;
        triggerGesture({
          type: 'doubleTap',
          position: { x: endX, y: endY },
          timestamp: now,
        });
      } else {
        // 单击
        touchState.current.tapCount = 1;
        touchState.current.lastTapTime = tapTime;
        
        // 延迟处理单击，等待可能的双击
        setTimeout(() => {
          if (touchState.current.tapCount === 1) {
            triggerGesture({
              type: 'tap',
              position: { x: endX, y: endY },
              timestamp: Date.now(),
            });
            touchState.current.tapCount = 0;
          }
        }, mergedConfig.doubleTapThreshold);
      }
    }
    
    // 触发释放事件
    triggerGesture({
      type: 'release',
      timestamp: now,
    });
    
    // 重置状态
    setIsTouching(false);
    touchState.current.isSwiping = false;
    touchState.current.isPinching = false;
    touchState.current.isRotating = false;
    touchState.current.touchCount = 0;
  }, [disabled, mergedConfig, triggerGesture, isTouching]);
  
  // 事件监听器
  useEffect(() => {
    if (!containerRef.current) return;
    
    const element = containerRef.current;
    const options = { passive: mergedConfig.passive };
    
    element.addEventListener('touchstart', handleTouchStart, options);
    element.addEventListener('touchmove', handleTouchMove, options);
    element.addEventListener('touchend', handleTouchEnd, options);
    element.addEventListener('touchcancel', handleTouchEnd, options);
    
    return () => {
      element.removeEventListener('touchstart', handleTouchStart);
      element.removeEventListener('touchmove', handleTouchMove);
      element.removeEventListener('touchend', handleTouchEnd);
      element.removeEventListener('touchcancel', handleTouchEnd);
      
      // 清理计时器
      if (touchState.current.longPressTimer) {
        clearTimeout(touchState.current.longPressTimer);
      }
    };
  }, [handleTouchStart, handleTouchMove, handleTouchEnd, mergedConfig.passive]);
  
  return (
    <div
      ref={containerRef}
      className={cn(
        'relative touch-manipulation',
        disabled && 'pointer-events-none opacity-50',
        className
      )}
      style={{
        touchAction: mergedConfig.preventDefault ? 'none' : 'auto',
        userSelect: 'none',
        WebkitUserSelect: 'none',
      }}
      {...props}
    >
      {children}
    </div>
  );
};

export default TouchGestureHandler;