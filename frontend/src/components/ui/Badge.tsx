'use client';

import { HTMLAttributes, forwardRef } from 'react';

type BadgeVariant = 'default' | 'success' | 'warning' | 'danger' | 'info';
type BadgeSize = 'sm' | 'md' | 'lg';

interface BadgeProps extends HTMLAttributes<HTMLSpanElement> {
  variant?: BadgeVariant;
  size?: BadgeSize;
  pulse?: boolean;
}

const Badge = forwardRef<HTMLSpanElement, BadgeProps>(
  ({ className = '', variant = 'default', size = 'md', pulse = false, children, ...props }, ref) => {
    const baseStyles = 'inline-flex items-center font-medium rounded-full';

    const variantStyles: Record<BadgeVariant, string> = {
      default: 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-200',
      success: 'bg-success-light text-success-dark dark:bg-success/20 dark:text-success',
      warning: 'bg-warning-light text-warning-dark dark:bg-warning/20 dark:text-warning',
      danger: 'bg-danger-light text-danger-dark dark:bg-danger/20 dark:text-danger',
      info: 'bg-brand-100 text-brand-800 dark:bg-brand-900/50 dark:text-brand-200',
    };

    const sizeStyles: Record<BadgeSize, string> = {
      sm: 'text-xs px-2 py-0.5',
      md: 'text-sm px-2.5 py-0.5',
      lg: 'text-base px-3 py-1',
    };

    const pulseStyles = pulse ? 'animate-pulse' : '';

    return (
      <span
        ref={ref}
        className={`${baseStyles} ${variantStyles[variant]} ${sizeStyles[size]} ${pulseStyles} ${className}`}
        {...props}
      >
        {children}
      </span>
    );
  }
);

Badge.displayName = 'Badge';

export { Badge };
export type { BadgeVariant, BadgeSize };
