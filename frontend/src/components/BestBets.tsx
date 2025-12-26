'use client';

import { motion } from 'framer-motion';
import { Star, TrendingUp, Clock, Target, Zap } from 'lucide-react';
import { ValueBet } from '@/lib/api';
import { Badge } from './ui/Badge';

interface BestBetsProps {
  bets: ValueBet[];
  isLoading?: boolean;
}

export function BestBets({ bets, isLoading }: BestBetsProps) {
  // Sort by edge and take top 3
  const topBets = [...bets]
    .sort((a, b) => b.edge - a.edge)
    .slice(0, 3);

  if (isLoading) {
    return (
      <div className="bg-surface-primary rounded-xl shadow-card border border-surface-border p-5">
        <div className="flex items-center gap-2 mb-4">
          <div className="w-5 h-5 bg-gray-200 dark:bg-gray-700 rounded animate-pulse" />
          <div className="h-5 bg-gray-200 dark:bg-gray-700 rounded w-24 animate-pulse" />
        </div>
        <div className="space-y-3">
          {[1, 2, 3].map((i) => (
            <div key={i} className="h-20 bg-surface-secondary rounded-lg animate-pulse" />
          ))}
        </div>
      </div>
    );
  }

  if (topBets.length === 0) {
    return (
      <div className="bg-surface-primary rounded-xl shadow-card border border-surface-border p-5">
        <div className="flex items-center gap-2 mb-4">
          <div className="p-1.5 bg-amber-100 dark:bg-amber-900/30 rounded-lg">
            <Star className="w-4 h-4 text-amber-600 dark:text-amber-400" />
          </div>
          <h2 className="text-lg font-semibold text-text-primary">Best Bets</h2>
        </div>
        <div className="text-center py-8">
          <Target className="w-10 h-10 mx-auto text-text-muted mb-2" />
          <p className="text-text-secondary">No high-confidence bets available</p>
          <p className="text-sm text-text-muted mt-1">Check back later for opportunities</p>
        </div>
      </div>
    );
  }

  const formatOdds = (odds: number) => odds > 0 ? `+${odds}` : odds.toString();
  const formatPercent = (value: number) => `${(value * 100).toFixed(1)}%`;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-surface-primary rounded-xl shadow-card border border-surface-border p-5"
    >
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <div className="p-1.5 bg-amber-100 dark:bg-amber-900/30 rounded-lg">
            <Star className="w-4 h-4 text-amber-600 dark:text-amber-400" />
          </div>
          <h2 className="text-lg font-semibold text-text-primary">Best Bets</h2>
        </div>
        <Badge variant="warning" size="sm">
          <Zap className="w-3 h-3 mr-1" />
          Top {topBets.length}
        </Badge>
      </div>

      <div className="space-y-3">
        {topBets.map((bet, index) => (
          <motion.div
            key={`${bet.game_id}-${bet.description}-${index}`}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: index * 0.1 }}
            className="relative bg-gradient-to-r from-amber-50 to-surface-secondary dark:from-amber-900/10 dark:to-surface-secondary rounded-lg p-4 border border-amber-200/50 dark:border-amber-800/30"
          >
            {/* Rank Badge */}
            <div className="absolute -left-2 -top-2 w-6 h-6 bg-amber-500 rounded-full flex items-center justify-center text-white text-xs font-bold shadow-md">
              {index + 1}
            </div>

            <div className="ml-2">
              {/* Description */}
              <h3 className="font-semibold text-text-primary mb-1">{bet.description}</h3>
              <p className="text-xs text-text-muted mb-2">{bet.bookmaker}</p>

              {/* Metrics Row */}
              <div className="flex items-center gap-4 text-sm">
                <div className="flex items-center gap-1">
                  <span className="text-text-muted">Odds:</span>
                  <span className="font-semibold text-text-primary">{formatOdds(bet.odds)}</span>
                </div>
                <div className="flex items-center gap-1 text-success">
                  <TrendingUp className="w-3.5 h-3.5" />
                  <span className="font-semibold">{formatPercent(bet.edge)} edge</span>
                </div>
                <div className="flex items-center gap-1 text-brand-600 dark:text-brand-400">
                  <Target className="w-3.5 h-3.5" />
                  <span className="font-medium">{formatPercent(bet.model_probability)}</span>
                </div>
              </div>
            </div>
          </motion.div>
        ))}
      </div>
    </motion.div>
  );
}
