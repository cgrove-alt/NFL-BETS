'use client';

import { motion } from 'framer-motion';
import { Clock, TrendingUp, Target, ChevronRight } from 'lucide-react';
import { ConfidenceCircle } from './ConfidenceGauge';
import { GameInfo } from '@/lib/api';
import { getTeamColors } from '@/lib/team-colors';
import { Badge } from './ui/Badge';

interface GameCardProps {
  game: GameInfo;
  onClick: () => void;
  isSelected?: boolean;
}

export function GameCard({ game, onClick, isSelected }: GameCardProps) {
  const kickoffDate = new Date(game.kickoff);
  const isToday = new Date().toDateString() === kickoffDate.toDateString();
  const isTomorrow = new Date(Date.now() + 86400000).toDateString() === kickoffDate.toDateString();

  const awayColors = getTeamColors(game.away_team);
  const homeColors = getTeamColors(game.home_team);

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString('en-US', {
      hour: 'numeric',
      minute: '2-digit',
      hour12: true
    });
  };

  const formatDate = (date: Date) => {
    if (isToday) return 'Today';
    if (isTomorrow) return 'Tomorrow';
    return date.toLocaleDateString('en-US', {
      weekday: 'short',
      month: 'short',
      day: 'numeric'
    });
  };

  const getUrgencyInfo = (count: number, edge: number | null) => {
    if (count === 0) return { color: 'bg-gray-200 dark:bg-gray-700', variant: 'default' as const };
    if (edge && edge >= 0.05) return { color: 'bg-danger', variant: 'danger' as const };
    if (edge && edge >= 0.03) return { color: 'bg-warning', variant: 'warning' as const };
    return { color: 'bg-success', variant: 'success' as const };
  };

  const urgency = getUrgencyInfo(game.value_bet_count, game.best_edge);
  const edgePercentage = game.best_edge ? `${(game.best_edge * 100).toFixed(1)}%` : null;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      whileHover={{ scale: 1.02 }}
      transition={{ duration: 0.2 }}
      onClick={onClick}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => e.key === 'Enter' && onClick()}
      aria-label={`${game.away_team} at ${game.home_team}, ${formatDate(kickoffDate)} at ${formatTime(kickoffDate)}. ${game.value_bet_count} value bets available.`}
      className={`
        bg-surface-primary rounded-xl shadow-card border-2 p-4 cursor-pointer
        transition-all duration-200 hover:shadow-card-hover
        focus-visible:ring-2 focus-visible:ring-brand-500 focus-visible:outline-none
        ${isSelected
          ? 'border-brand-500 ring-2 ring-brand-200 dark:ring-brand-800'
          : 'border-surface-border hover:border-brand-300 dark:hover:border-brand-700'}
      `}
    >
      {/* Teams and Time */}
      <div className="flex justify-between items-start mb-3">
        <div className="flex-1">
          <div className="flex items-center gap-2 mb-1.5">
            {/* Away Team Badge */}
            <div
              className="px-2 py-0.5 rounded text-xs font-bold text-white"
              style={{ backgroundColor: awayColors.primary }}
            >
              {game.away_team}
            </div>
            <span className="text-text-muted text-sm">@</span>
            {/* Home Team Badge */}
            <div
              className="px-2 py-0.5 rounded text-xs font-bold text-white"
              style={{ backgroundColor: homeColors.primary }}
            >
              {game.home_team}
            </div>
          </div>
          <div className="flex items-center gap-1.5 text-sm text-text-secondary">
            <Clock className="w-3.5 h-3.5" />
            <span>{formatDate(kickoffDate)} • {formatTime(kickoffDate)}</span>
          </div>
        </div>

        {/* Confidence Circle */}
        {game.model_confidence && game.value_bet_count > 0 && (
          <ConfidenceCircle value={game.model_confidence} size={50} />
        )}
      </div>

      {/* Value Bet Indicator Bar */}
      <div className="mb-3">
        <div className="flex justify-between items-center text-xs mb-1.5">
          <span className="text-text-secondary flex items-center gap-1">
            <Target className="w-3.5 h-3.5" />
            Value Bets
          </span>
          <Badge variant={urgency.variant} size="sm">
            {game.value_bet_count}
          </Badge>
        </div>
        <div className="w-full bg-gray-100 dark:bg-gray-800 rounded-full h-2 overflow-hidden">
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: `${Math.min(game.value_bet_count * 20, 100)}%` }}
            transition={{ duration: 0.5, delay: 0.2 }}
            className={`h-2 rounded-full ${urgency.color}`}
          />
        </div>
      </div>

      {/* Best Edge Preview */}
      {game.value_bet_count > 0 && game.best_bet_description ? (
        <div className="bg-success-light dark:bg-success/10 rounded-lg p-3 border border-success/20">
          <div className="flex justify-between items-center">
            <span className="text-sm text-text-primary flex-1 mr-2">{game.best_bet_description}</span>
            <div className="flex items-center gap-1">
              <TrendingUp className="w-4 h-4 text-success" />
              <span className="text-sm font-bold text-success">{edgePercentage}</span>
            </div>
          </div>
        </div>
      ) : (
        <div className="bg-surface-secondary rounded-lg p-3 border border-surface-border">
          <span className="text-sm text-text-muted">No value bets found</span>
        </div>
      )}

      {/* Model vs Vegas (if available) */}
      {game.model_prediction !== null && game.vegas_line !== null && (
        <div className="mt-3 pt-3 border-t border-surface-border flex justify-between items-center text-xs">
          <div className="flex gap-4">
            <div>
              <span className="text-text-muted">Model: </span>
              <span className="font-semibold text-brand-600 dark:text-brand-400">
                {game.model_prediction > 0 ? '+' : ''}{game.model_prediction.toFixed(1)}
              </span>
            </div>
            <div>
              <span className="text-text-muted">Vegas: </span>
              <span className="font-semibold text-text-primary">
                {game.vegas_line > 0 ? '+' : ''}{game.vegas_line.toFixed(1)}
              </span>
            </div>
          </div>
          <ChevronRight className="w-4 h-4 text-text-muted" />
        </div>
      )}
    </motion.div>
  );
}

// Loading skeleton
export function GameCardSkeleton() {
  return (
    <div className="bg-surface-primary rounded-xl shadow-card border border-surface-border p-4 animate-pulse">
      <div className="flex justify-between items-start mb-3">
        <div className="flex-1">
          <div className="flex gap-2 mb-2">
            <div className="h-5 bg-gray-200 dark:bg-gray-700 rounded w-12" />
            <div className="h-5 bg-gray-200 dark:bg-gray-700 rounded w-4" />
            <div className="h-5 bg-gray-200 dark:bg-gray-700 rounded w-12" />
          </div>
          <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-28" />
        </div>
        <div className="w-12 h-12 bg-gray-200 dark:bg-gray-700 rounded-full" />
      </div>
      <div className="mb-3">
        <div className="flex justify-between mb-1.5">
          <div className="h-3 bg-gray-200 dark:bg-gray-700 rounded w-20" />
          <div className="h-5 bg-gray-200 dark:bg-gray-700 rounded-full w-8" />
        </div>
        <div className="h-2 bg-gray-200 dark:bg-gray-700 rounded-full" />
      </div>
      <div className="h-12 bg-surface-secondary rounded-lg" />
    </div>
  );
}
