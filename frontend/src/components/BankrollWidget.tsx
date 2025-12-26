'use client';

import { motion } from 'framer-motion';
import { Wallet, TrendingUp, TrendingDown, Clock, DollarSign, PiggyBank, AlertCircle } from 'lucide-react';
import { BankrollSummary } from '@/lib/api';

interface BankrollWidgetProps {
  bankroll: BankrollSummary | null;
  isLoading?: boolean;
}

export default function BankrollWidget({ bankroll, isLoading }: BankrollWidgetProps) {
  if (isLoading) {
    return (
      <div className="bg-surface-primary rounded-xl shadow-card border border-surface-border p-5 animate-pulse">
        <div className="flex items-center gap-2 mb-4">
          <div className="w-5 h-5 bg-gray-200 dark:bg-gray-700 rounded" />
          <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-20" />
        </div>
        <div className="h-9 bg-gray-200 dark:bg-gray-700 rounded w-32 mb-2" />
        <div className="h-5 bg-gray-200 dark:bg-gray-700 rounded w-40 mb-4" />
        <div className="grid grid-cols-2 gap-3">
          <div className="h-16 bg-surface-secondary rounded-lg" />
          <div className="h-16 bg-surface-secondary rounded-lg" />
        </div>
      </div>
    );
  }

  if (!bankroll) {
    return (
      <div className="bg-surface-primary rounded-xl shadow-card border border-surface-border p-5">
        <div className="flex items-center gap-2 text-text-muted">
          <AlertCircle className="w-5 h-5" />
          <p>Unable to load bankroll data</p>
        </div>
      </div>
    );
  }

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(value);
  };

  const formatPercent = (value: number) => {
    const sign = value >= 0 ? '+' : '';
    return `${sign}${(value * 100).toFixed(1)}%`;
  };

  const isProfit = bankroll.total_profit >= 0;
  const TrendIcon = isProfit ? TrendingUp : TrendingDown;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-surface-primary rounded-xl shadow-card border border-surface-border p-5"
    >
      {/* Header */}
      <div className="flex items-center gap-2 mb-3">
        <div className="p-1.5 bg-brand-100 dark:bg-brand-900/30 rounded-lg">
          <Wallet className="w-4 h-4 text-brand-600 dark:text-brand-400" />
        </div>
        <h3 className="text-sm font-medium text-text-secondary">Bankroll</h3>
      </div>

      {/* Main Balance */}
      <motion.div
        key={bankroll.current_bankroll}
        initial={{ scale: 0.95 }}
        animate={{ scale: 1 }}
        className="text-3xl font-bold text-text-primary mb-1"
      >
        {formatCurrency(bankroll.current_bankroll)}
      </motion.div>

      {/* Profit/Loss with Icon */}
      <div className={`flex items-center gap-1.5 text-sm font-medium mb-4 ${
        isProfit ? 'text-success' : 'text-danger'
      }`}>
        <TrendIcon className="w-4 h-4" />
        <span>{formatCurrency(Math.abs(bankroll.total_profit))}</span>
        <span className={`px-1.5 py-0.5 rounded text-xs ${
          isProfit
            ? 'bg-success-light dark:bg-success/20'
            : 'bg-danger-light dark:bg-danger/20'
        }`}>
          {formatPercent(bankroll.roi)} ROI
        </span>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-2 gap-3">
        <div className="bg-surface-secondary rounded-lg p-3">
          <div className="flex items-center gap-1.5 text-xs text-text-muted mb-1">
            <PiggyBank className="w-3.5 h-3.5" />
            Available
          </div>
          <p className="font-semibold text-text-primary">
            {formatCurrency(bankroll.available_bankroll)}
          </p>
        </div>
        <div className="bg-warning-light dark:bg-warning/10 rounded-lg p-3">
          <div className="flex items-center gap-1.5 text-xs text-warning-dark dark:text-warning mb-1">
            <Clock className="w-3.5 h-3.5" />
            Pending
          </div>
          <p className="font-semibold text-warning-dark dark:text-warning">
            {formatCurrency(bankroll.pending_exposure)}
          </p>
        </div>
      </div>
    </motion.div>
  );
}
