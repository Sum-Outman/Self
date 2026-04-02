import React, { useEffect, useRef } from 'react';
import { X } from 'lucide-react';
import { cn } from '../../utils/cn';
import { Button } from './Button';

export interface ModalProps {
  /** 是否显示模态框 */
  isOpen: boolean;
  /** 关闭模态框的回调函数 */
  onClose: () => void;
  /** 模态框标题 */
  title?: string;
  /** 模态框描述 */
  description?: string;
  /** 模态框内容 */
  children: React.ReactNode;
  /** 确认按钮文字 */
  confirmText?: string;
  /** 取消按钮文字 */
  cancelText?: string;
  /** 确认按钮点击回调 */
  onConfirm?: () => void;
  /** 取消按钮点击回调 */
  onCancel?: () => void;
  /** 是否显示确认按钮 */
  showConfirm?: boolean;
  /** 是否显示取消按钮 */
  showCancel?: boolean;
  /** 确认按钮是否加载中 */
  confirmLoading?: boolean;
  /** 是否禁用ESC键关闭 */
  disableEscapeClose?: boolean;
  /** 是否禁用点击遮罩层关闭 */
  disableOverlayClose?: boolean;
  /** 模态框大小 */
  size?: 'sm' | 'md' | 'lg' | 'xl' | 'full';
  /** 自定义类名 */
  className?: string;
}

const sizeClasses = {
  sm: 'max-w-sm',
  md: 'max-w-md',
  lg: 'max-w-lg',
  xl: 'max-w-xl',
  full: 'max-w-full mx-4',
};

export const Modal: React.FC<ModalProps> = ({
  isOpen,
  onClose,
  title,
  description,
  children,
  confirmText = '确认',
  cancelText = '取消',
  onConfirm,
  onCancel,
  showConfirm = true,
  showCancel = true,
  confirmLoading = false,
  disableEscapeClose = false,
  disableOverlayClose = false,
  size = 'md',
  className,
}) => {
  const modalRef = useRef<HTMLDivElement>(null);
  
  // 处理ESC键关闭
  useEffect(() => {
    const handleEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && isOpen && !disableEscapeClose) {
        onClose();
      }
    };
    
    document.addEventListener('keydown', handleEscape);
    return () => document.removeEventListener('keydown', handleEscape);
  }, [isOpen, onClose, disableEscapeClose]);
  
  // 处理点击遮罩层关闭
  const handleOverlayClick = (event: React.MouseEvent<HTMLDivElement>) => {
    if (event.target === event.currentTarget && !disableOverlayClose) {
      onClose();
    }
  };
  
  // 处理确认按钮点击
  const handleConfirm = () => {
    if (onConfirm) {
      onConfirm();
    } else {
      onClose();
    }
  };
  
  // 处理取消按钮点击
  const handleCancel = () => {
    if (onCancel) {
      onCancel();
    } else {
      onClose();
    }
  };
  
  // 如果模态框未打开，不渲染任何内容
  if (!isOpen) return null;
  
  return (
    <div className="fixed inset-0 z-50 overflow-y-auto">
      {/* 遮罩层 */}
      <div
        className="fixed inset-0 bg-black bg-opacity-50 transition-opacity"
        onClick={handleOverlayClick}
        aria-hidden="true"
      />
      
      <div className="flex min-h-full items-center justify-center p-4">
        <div
          ref={modalRef}
          className={cn(
            'relative w-full transform overflow-hidden rounded-lg bg-white dark:bg-gray-800 shadow-xl transition-all',
            sizeClasses[size],
            className
          )}
          role="dialog"
          aria-modal="true"
          aria-labelledby={title ? 'modal-title' : undefined}
          aria-describedby={description ? 'modal-description' : undefined}
        >
          {/* 头部 */}
          {(title || description) && (
            <div className="px-6 py-4 border-b border-gray-200 dark:border-gray-700">
              <div className="flex items-center justify-between">
                <div>
                  {title && (
                    <h3
                      id="modal-title"
                      className="text-lg font-semibold text-gray-900 dark:text-gray-100"
                    >
                      {title}
                    </h3>
                  )}
                  {description && (
                    <p
                      id="modal-description"
                      className="mt-1 text-sm text-gray-500 dark:text-gray-400"
                    >
                      {description}
                    </p>
                  )}
                </div>
                <button
                  type="button"
                  className="ml-4 text-gray-400 hover:text-gray-500 focus:outline-none focus:ring-2 focus:ring-gray-500 rounded-lg p-1"
                  onClick={onClose}
                  aria-label="关闭"
                >
                  <X className="h-5 w-5" />
                </button>
              </div>
            </div>
          )}
          
          {/* 内容 */}
          <div className="px-6 py-4">
            {children}
          </div>
          
          {/* 底部按钮 */}
          {(showConfirm || showCancel) && (
            <div className="px-6 py-4 border-t border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900/50 flex justify-end space-x-3">
              {showCancel && (
                <Button
                  variant="secondary"
                  onClick={handleCancel}
                  disabled={confirmLoading}
                >
                  {cancelText}
                </Button>
              )}
              
              {showConfirm && (
                <Button
                  variant="primary"
                  onClick={handleConfirm}
                  loading={confirmLoading}
                  disabled={confirmLoading}
                >
                  {confirmText}
                </Button>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

Modal.displayName = 'Modal';

// 导出预定义的模态框组件
export const AlertModal = (props: Omit<ModalProps, 'showConfirm' | 'showCancel'>) => (
  <Modal showConfirm showCancel={false} {...props} />
);

export const ConfirmModal = (props: Omit<ModalProps, 'showConfirm' | 'showCancel'>) => (
  <Modal showConfirm showCancel {...props} />
);

export const DialogModal = (props: Omit<ModalProps, 'showConfirm' | 'showCancel'>) => (
  <Modal showConfirm={false} showCancel={false} {...props} />
);