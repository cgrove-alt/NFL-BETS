'use client';

import { forwardRef, HTMLAttributes } from 'react';
import { motion, HTMLMotionProps } from 'framer-motion';

interface CardProps extends HTMLAttributes<HTMLDivElement> {
  variant?: 'default' | 'elevated' | 'outlined';
  hover?: boolean;
  animate?: boolean;
}

const Card = forwardRef<HTMLDivElement, CardProps>(
  ({ className = '', variant = 'default', hover = true, animate = true, children, ...props }, ref) => {
    const baseStyles = 'rounded-xl transition-all duration-200';

    const variantStyles = {
      default: 'bg-surface-primary border border-surface-border shadow-card',
      elevated: 'bg-surface-elevated shadow-elevated',
      outlined: 'bg-transparent border-2 border-surface-border',
    };

    const hoverStyles = hover
      ? 'hover:shadow-card-hover hover:border-brand-200 dark:hover:border-brand-700'
      : '';

    const combinedClassName = `${baseStyles} ${variantStyles[variant]} ${hoverStyles} ${className}`;

    if (animate) {
      return (
        <motion.div
          ref={ref}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
          className={combinedClassName}
          {...(props as HTMLMotionProps<'div'>)}
        >
          {children}
        </motion.div>
      );
    }

    return (
      <div ref={ref} className={combinedClassName} {...props}>
        {children}
      </div>
    );
  }
);

Card.displayName = 'Card';

export { Card };
