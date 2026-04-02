import React from 'react';
import { cn } from '../../utils/cn';
import { useMobileDetect } from '../../hooks/useMobileDetect';

export interface ResponsiveContainerProps extends Omit<React.HTMLAttributes<HTMLDivElement>, 'hidden'> {
  /** 容器类型 */
  type?: 'section' | 'div' | 'main' | 'article' | 'header' | 'footer';
  /** 最大宽度限制 */
  maxWidth?: 'none' | 'sm' | 'md' | 'lg' | 'xl' | '2xl' | 'full' | 'screen';
  /** 内边距 */
  padding?: 'none' | 'sm' | 'md' | 'lg' | 'xl' | 'responsive';
  /** 外边距 */
  margin?: 'none' | 'auto' | 'responsive';
  /** 是否启用滚动 */
  scrollable?: boolean;
  /** 背景颜色 */
  background?: 'transparent' | 'light' | 'dark' | 'white' | 'gray';
  /** 边框样式 */
  border?: 'none' | 'light' | 'medium' | 'heavy' | 'rounded';
  /** 阴影样式 */
  shadow?: 'none' | 'sm' | 'md' | 'lg' | 'xl' | 'responsive';
  /** 是否启用网格布局 */
  grid?: boolean;
  /** 网格列数（响应式） */
  gridCols?: {
    default?: number;
    sm?: number;
    md?: number;
    lg?: number;
    xl?: number;
    '2xl'?: number;
  };
  /** 网格间距 */
  gap?: 'none' | 'sm' | 'md' | 'lg' | 'xl' | 'responsive';
  /** 是否启用弹性布局 */
  flex?: boolean;
  /** 弹性布局方向 */
  flexDirection?: 'row' | 'col' | 'row-reverse' | 'col-reverse' | 'responsive';
  /** 弹性布局对齐方式 */
  alignItems?: 'start' | 'center' | 'end' | 'stretch' | 'baseline' | 'responsive';
  /** 弹性布局分布方式 */
  justifyContent?: 'start' | 'center' | 'end' | 'between' | 'around' | 'evenly' | 'responsive';
  /** 是否隐藏在某些断点 */
  hidden?: {
    sm?: boolean;
    md?: boolean;
    lg?: boolean;
    xl?: boolean;
    '2xl'?: boolean;
  };
  /** 是否只在某些断点显示 */
  visible?: {
    sm?: boolean;
    md?: boolean;
    lg?: boolean;
    xl?: boolean;
    '2xl'?: boolean;
  };
  /** 子元素 */
  children?: React.ReactNode;
}

const maxWidthClasses = {
  none: '',
  sm: 'max-w-sm',
  md: 'max-w-md',
  lg: 'max-w-lg',
  xl: 'max-w-xl',
  '2xl': 'max-w-2xl',
  full: 'max-w-full',
  screen: 'max-w-screen',
};

const paddingClasses = {
  none: 'p-0',
  sm: 'p-4',
  md: 'p-6',
  lg: 'p-8',
  xl: 'p-10',
  responsive: 'p-4 md:p-6 lg:p-8',
};

const marginClasses = {
  none: 'm-0',
  auto: 'mx-auto',
  responsive: 'mx-4 md:mx-6 lg:mx-auto',
};

const backgroundClasses = {
  transparent: 'bg-transparent',
  light: 'bg-gray-50 dark:bg-gray-900',
  dark: 'bg-gray-900 dark:bg-gray-950',
  white: 'bg-white dark:bg-gray-800',
  gray: 'bg-gray-100 dark:bg-gray-800',
};

const borderClasses = {
  none: 'border-0',
  light: 'border border-gray-200 dark:border-gray-700',
  medium: 'border-2 border-gray-300 dark:border-gray-600',
  heavy: 'border-4 border-gray-400 dark:border-gray-500',
  rounded: 'border border-gray-200 dark:border-gray-700 rounded-lg',
};

const shadowClasses = {
  none: '',
  sm: 'shadow-sm',
  md: 'shadow-md',
  lg: 'shadow-lg',
  xl: 'shadow-xl',
  responsive: 'shadow-sm md:shadow-md lg:shadow-lg',
};

const gapClasses = {
  none: 'gap-0',
  sm: 'gap-2',
  md: 'gap-4',
  lg: 'gap-6',
  xl: 'gap-8',
  responsive: 'gap-2 md:gap-4 lg:gap-6',
};

export const ResponsiveContainer: React.FC<ResponsiveContainerProps> = ({
  type = 'div',
  maxWidth = 'xl',
  padding = 'responsive',
  margin = 'auto',
  scrollable = false,
  background = 'transparent',
  border = 'none',
  shadow = 'none',
  grid = false,
  gridCols = { default: 1, md: 2, lg: 3 },
  gap = 'responsive',
  flex = false,
  flexDirection = 'col',
  alignItems = 'stretch',
  justifyContent = 'start',
  hidden,
  visible,
  className,
  children,
  ...props
}) => {
  const { isMobile } = useMobileDetect();
  
  // 构建响应式类名
  const classes = cn(
    // 基础样式
    'w-full',
    
    // 最大宽度
    maxWidthClasses[maxWidth],
    
    // 内边距
    paddingClasses[padding],
    
    // 外边距
    marginClasses[margin],
    
    // 背景
    backgroundClasses[background],
    
    // 边框
    borderClasses[border],
    
    // 阴影
    shadowClasses[shadow],
    
    // 滚动
    scrollable && 'overflow-auto',
    
    // 网格布局
    grid && 'grid',
    grid && getGridColsClass(gridCols),
    grid && gapClasses[gap],
    
    // 弹性布局
    flex && 'flex',
    flex && getFlexDirectionClass(flexDirection, isMobile),
    flex && getAlignItemsClass(alignItems, isMobile),
    flex && getJustifyContentClass(justifyContent, isMobile),
    
    // 响应式隐藏/显示
    hidden && getHiddenClasses(hidden),
    visible && getVisibleClasses(visible),
    
    // 自定义类名
    className
  );
  
  const Container = type;
  
  return (
    <Container className={classes} {...props}>
      {children}
    </Container>
  );
};

ResponsiveContainer.displayName = 'ResponsiveContainer';

// 辅助函数：获取网格列数类名
function getGridColsClass(gridCols: ResponsiveContainerProps['gridCols']): string {
  const classes: string[] = [];
  
  if (gridCols?.default) {
    classes.push(`grid-cols-${gridCols.default}`);
  }
  
  if (gridCols?.sm) {
    classes.push(`sm:grid-cols-${gridCols.sm}`);
  }
  
  if (gridCols?.md) {
    classes.push(`md:grid-cols-${gridCols.md}`);
  }
  
  if (gridCols?.lg) {
    classes.push(`lg:grid-cols-${gridCols.lg}`);
  }
  
  if (gridCols?.xl) {
    classes.push(`xl:grid-cols-${gridCols.xl}`);
  }
  
  if (gridCols?.['2xl']) {
    classes.push(`2xl:grid-cols-${gridCols['2xl']}`);
  }
  
  return classes.join(' ');
}

// 辅助函数：获取弹性布局方向类名
function getFlexDirectionClass(
  direction: ResponsiveContainerProps['flexDirection'],
  isMobile: boolean
): string {
  if (direction === 'responsive') {
    return isMobile ? 'flex-col' : 'flex-row';
  }
  
  return direction === 'col' ? 'flex-col' :
         direction === 'row' ? 'flex-row' :
         direction === 'col-reverse' ? 'flex-col-reverse' :
         direction === 'row-reverse' ? 'flex-row-reverse' : 'flex-col';
}

// 辅助函数：获取对齐方式类名
function getAlignItemsClass(
  align: ResponsiveContainerProps['alignItems'],
  isMobile: boolean
): string {
  if (align === 'responsive') {
    return isMobile ? 'items-stretch' : 'items-center';
  }
  
  return align === 'start' ? 'items-start' :
         align === 'center' ? 'items-center' :
         align === 'end' ? 'items-end' :
         align === 'stretch' ? 'items-stretch' :
         align === 'baseline' ? 'items-baseline' : 'items-stretch';
}

// 辅助函数：获取分布方式类名
function getJustifyContentClass(
  justify: ResponsiveContainerProps['justifyContent'],
  isMobile: boolean
): string {
  if (justify === 'responsive') {
    return isMobile ? 'justify-start' : 'justify-between';
  }
  
  return justify === 'start' ? 'justify-start' :
         justify === 'center' ? 'justify-center' :
         justify === 'end' ? 'justify-end' :
         justify === 'between' ? 'justify-between' :
         justify === 'around' ? 'justify-around' :
         justify === 'evenly' ? 'justify-evenly' : 'justify-start';
}

// 辅助函数：获取隐藏类名
function getHiddenClasses(hidden: ResponsiveContainerProps['hidden']): string {
  const classes: string[] = [];
  
  if (hidden?.sm) {
    classes.push('hidden sm:block');
  }
  
  if (hidden?.md) {
    classes.push('hidden md:block');
  }
  
  if (hidden?.lg) {
    classes.push('hidden lg:block');
  }
  
  if (hidden?.xl) {
    classes.push('hidden xl:block');
  }
  
  if (hidden?.['2xl']) {
    classes.push('hidden 2xl:block');
  }
  
  return classes.join(' ');
}

// 辅助函数：获取显示类名
function getVisibleClasses(visible: ResponsiveContainerProps['visible']): string {
  const classes: string[] = [];
  
  if (visible?.sm) {
    classes.push('block sm:hidden');
  }
  
  if (visible?.md) {
    classes.push('block md:hidden');
  }
  
  if (visible?.lg) {
    classes.push('block lg:hidden');
  }
  
  if (visible?.xl) {
    classes.push('block xl:hidden');
  }
  
  if (visible?.['2xl']) {
    classes.push('block 2xl:hidden');
  }
  
  return classes.join(' ');
}

// 预定义的响应式容器组件
export const MobileContainer = (props: Omit<ResponsiveContainerProps, 'maxWidth' | 'padding'>) => (
  <ResponsiveContainer
    maxWidth="full"
    padding="sm"
    {...props}
  />
);

export const TabletContainer = (props: Omit<ResponsiveContainerProps, 'maxWidth' | 'padding'>) => (
  <ResponsiveContainer
    maxWidth="lg"
    padding="md"
    {...props}
  />
);

export const DesktopContainer = (props: Omit<ResponsiveContainerProps, 'maxWidth' | 'padding'>) => (
  <ResponsiveContainer
    maxWidth="xl"
    padding="lg"
    {...props}
  />
);

export const SectionContainer = (props: Omit<ResponsiveContainerProps, 'type' | 'maxWidth'>) => (
  <ResponsiveContainer
    type="section"
    maxWidth="screen"
    {...props}
  />
);

export const MainContainer = (props: Omit<ResponsiveContainerProps, 'type' | 'maxWidth'>) => (
  <ResponsiveContainer
    type="main"
    maxWidth="screen"
    {...props}
  />
);