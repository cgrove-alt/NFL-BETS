'use client';

import { HTMLAttributes } from 'react';

interface SkeletonProps extends HTMLAttributes<HTMLDivElement> {
  variant?: 'text' | 'circular' | 'rectangular';
  width?: string | number;
  height?: string | number;
}

export function Skeleton({
  className = '',
  variant = 'rectangular',
  width,
  height,
  style,
  ...props
}: SkeletonProps) {
  const baseStyles = 'animate-pulse bg-gray-200 dark:bg-gray-700';

  const variantStyles = {
    text: 'rounded',
    circular: 'rounded-full',
    rectangular: 'rounded-lg',
  };

  const combinedStyles = {
    width: width ?? (variant === 'text' ? '100%' : undefined),
    height: height ?? (variant === 'text' ? '1em' : undefined),
    ...style,
  };

  return (
    <div
      className={`${baseStyles} ${variantStyles[variant]} ${className}`}
      style={combinedStyles}
      {...props}
    />
  );
}

// Pre-built skeleton components for common use cases
export function SkeletonCard() {
  return (
    <div className="p-4 rounded-xl border border-surface-border bg-surface-primary">
      <div className="flex items-center gap-4 mb-4">
        <Skeleton variant="circular" width={48} height={48} />
        <div className="flex-1 space-y-2">
          <Skeleton variant="text" width="60%" height={20} />
          <Skeleton variant="text" width="40%" height={16} />
        </div>
      </div>
      <Skeleton variant="rectangular" width="100%" height={80} />
    </div>
  );
}

export function SkeletonValueBet() {
  return (
    <div className="p-4 rounded-xl border border-surface-border bg-surface-primary border-l-4 border-l-gray-300">
      <div className="flex justify-between items-start mb-3">
        <Skeleton variant="text" width="50%" height={24} />
        <Skeleton variant="rectangular" width={60} height={24} className="rounded-full" />
      </div>
      <div className="grid grid-cols-3 gap-4">
        <div className="space-y-1">
          <Skeleton variant="text" width="80%" height={14} />
          <Skeleton variant="text" width="60%" height={20} />
        </div>
        <div className="space-y-1">
          <Skeleton variant="text" width="80%" height={14} />
          <Skeleton variant="text" width="60%" height={20} />
        </div>
        <div className="space-y-1">
          <Skeleton variant="text" width="80%" height={14} />
          <Skeleton variant="text" width="60%" height={20} />
        </div>
      </div>
    </div>
  );
}

export function SkeletonBankroll() {
  return (
    <div className="p-6 rounded-xl border border-surface-border bg-surface-primary">
      <div className="flex items-center gap-2 mb-4">
        <Skeleton variant="circular" width={24} height={24} />
        <Skeleton variant="text" width={100} height={20} />
      </div>
      <Skeleton variant="text" width="70%" height={36} className="mb-2" />
      <Skeleton variant="text" width="40%" height={16} />
    </div>
  );
}
