'use client';

import { motion } from 'framer-motion';
import { TrendingUp, DollarSign, Target, Clock, Percent, Calculator } from 'lucide-react';
import { ValueBet } from '@/lib/api';
import { Badge, BadgeVariant } from './ui/Badge';

interface ValueBetCardProps {
  bet: ValueBet;
}

const urgencyConfig: Record<string, { variant: BadgeVariant; pulse: boolean }> = {
  CRITICAL: { variant: 'danger', pulse: true },
  HIGH: { variant: 'warning', pulse: false },
  MEDIUM: { variant: 'info', pulse: false },
  LOW: { variant: 'default', pulse: false },
};

const getBorderColor = (urgency: string) => {
  switch (urgency) {
    case 'CRITICAL': return 'border-l-danger';
    case 'HIGH': return 'border-l-warning';
    case 'MEDIUM': return 'border-l-brand-500';
    default: return 'border-l-gray-400';
  }
};

export default function ValueBetCard({ bet }: ValueBetCardProps) {
  const formatOdds = (odds: number) => {
    return odds > 0 ? `+${odds}` : odds.toString();
  };

  const formatPercent = (value: number) => {
    return `${(value * 100).toFixed(1)}%`;
  };

  const formatCurrency = (value: number | null) => {
    if (value === null) return 'N/A';
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(value);
  };

  const urgency = urgencyConfig[bet.urgency] || urgencyConfig.LOW;

  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      whileHover={{ scale: 1.01, boxShadow: '0 10px 40px rgba(0,0,0,0.1)' }}
      transition={{ duration: 0.2 }}
      className={`
        bg-surface-primary rounded-xl shadow-card p-4
        border-l-4 ${getBorderColor(bet.urgency)}
        border border-surface-border
        hover:border-brand-300 dark:hover:border-brand-700
        transition-colors duration-200
      `}
    >
      {/* Header */}
      <div className="flex justify-between items-start mb-3">
        <div className="flex-1 min-w-0">
          <h3 className="font-semibold text-lg text-text-primary truncate">{bet.description}</h3>
          <p className="text-sm text-text-secondary">{bet.bookmaker}</p>
        </div>
        <Badge variant={urgency.variant} pulse={urgency.pulse} size="sm">
          {bet.urgency}
        </Badge>
      </div>

      {/* Primary Metrics */}
      <div className="grid grid-cols-3 gap-3 mb-3">
        <div className="bg-surface-secondary rounded-lg p-2.5">
          <div className="flex items-center gap-1 text-xs text-text-muted mb-1">
            <Calculator className="w-3 h-3" />
            Odds
          </div>
          <p className="font-semibold text-text-primary">{formatOdds(bet.odds)}</p>
        </div>
        <div className="bg-success-light dark:bg-success/10 rounded-lg p-2.5">
          <div className="flex items-center gap-1 text-xs text-success-dark dark:text-success mb-1">
            <TrendingUp className="w-3 h-3" />
            Edge
          </div>
          <p className="font-semibold text-success">{formatPercent(bet.edge)}</p>
        </div>
        <div className="bg-success-light dark:bg-success/10 rounded-lg p-2.5">
          <div className="flex items-center gap-1 text-xs text-success-dark dark:text-success mb-1">
            <Percent className="w-3 h-3" />
            EV
          </div>
          <p className="font-semibold text-success">{formatPercent(bet.expected_value)}</p>
        </div>
      </div>

      {/* Secondary Metrics */}
      <div className="grid grid-cols-2 gap-3">
        <div className="flex items-center gap-2 bg-surface-secondary rounded-lg p-2.5">
          <Target className="w-4 h-4 text-text-muted" />
          <div>
            <p className="text-xs text-text-muted">Model Prob</p>
            <p className="font-medium text-text-primary">{formatPercent(bet.model_probability)}</p>
          </div>
        </div>
        <div className="flex items-center gap-2 bg-brand-50 dark:bg-brand-900/20 rounded-lg p-2.5">
          <DollarSign className="w-4 h-4 text-brand-600 dark:text-brand-400" />
          <div>
            <p className="text-xs text-brand-600 dark:text-brand-400">Rec. Stake</p>
            <p className="font-semibold text-brand-700 dark:text-brand-300">{formatCurrency(bet.recommended_stake)}</p>
          </div>
        </div>
      </div>

      {/* Footer */}
      <div className="mt-3 pt-3 border-t border-surface-border flex items-center gap-1.5 text-xs text-text-muted">
        <Clock className="w-3 h-3" />
        <span>Detected {new Date(bet.detected_at).toLocaleString()}</span>
      </div>
    </motion.div>
  );
}
