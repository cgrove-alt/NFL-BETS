'use client';

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  TrendingUp,
  DollarSign,
  Users,
  Target,
  ChevronDown,
  ArrowUpDown,
  Clock,
  Calculator,
  Percent,
  User,
} from 'lucide-react';
import { ValueBet, GameInfo, GamePredictions, SpreadPrediction, PropPrediction } from '@/lib/api';
import { Badge } from './ui/Badge';
import { getTeamColors } from '@/lib/team-colors';

interface GameBetsDisplayProps {
  game: GameInfo;
  valueBets: ValueBet[];
  predictions: GamePredictions | null;
  isLoading?: boolean;
}

type BetCategory = 'spread' | 'moneyline' | 'totals' | 'props';

const categoryConfig: Record<BetCategory, { label: string; icon: React.ElementType; color: string }> = {
  spread: { label: 'Spread', icon: ArrowUpDown, color: 'text-brand-600 dark:text-brand-400' },
  moneyline: { label: 'Moneyline', icon: DollarSign, color: 'text-success' },
  totals: { label: 'Totals', icon: Target, color: 'text-warning' },
  props: { label: 'Player Props', icon: Users, color: 'text-purple-600 dark:text-purple-400' },
};

function categorizeBets(bets: ValueBet[]): Record<BetCategory, ValueBet[]> {
  const categorized: Record<BetCategory, ValueBet[]> = {
    spread: [],
    moneyline: [],
    totals: [],
    props: [],
  };

  bets.forEach((bet) => {
    const type = bet.bet_type.toLowerCase();
    if (type.includes('spread')) {
      categorized.spread.push(bet);
    } else if (type.includes('moneyline') || type.includes('ml')) {
      categorized.moneyline.push(bet);
    } else if (type.includes('total') || type.includes('over') || type.includes('under')) {
      categorized.totals.push(bet);
    } else {
      // Everything else is a prop bet
      categorized.props.push(bet);
    }
  });

  return categorized;
}

export function GameBetsDisplay({ game, valueBets, predictions, isLoading }: GameBetsDisplayProps) {
  const [expandedCategories, setExpandedCategories] = useState<Set<BetCategory>>(
    new Set(['spread', 'moneyline', 'props'])
  );

  const toggleCategory = (category: BetCategory) => {
    setExpandedCategories((prev) => {
      const next = new Set(prev);
      if (next.has(category)) {
        next.delete(category);
      } else {
        next.add(category);
      }
      return next;
    });
  };

  const categorizedBets = categorizeBets(valueBets);
  const homeColors = getTeamColors(game.home_team);
  const awayColors = getTeamColors(game.away_team);

  const formatOdds = (odds: number) => (odds > 0 ? `+${odds}` : odds.toString());
  const formatPercent = (value: number) => `${(value * 100).toFixed(1)}%`;
  const formatCurrency = (value: number | null) => {
    if (value === null) return 'N/A';
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(value);
  };

  if (isLoading) {
    return (
      <div className="space-y-4">
        {[1, 2, 3].map((i) => (
          <div key={i} className="h-24 bg-surface-secondary rounded-xl animate-pulse" />
        ))}
      </div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-4"
    >
      {/* Game Header */}
      <div className="bg-surface-primary rounded-xl border border-surface-border p-4">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-3">
            <div
              className="px-3 py-1 rounded-lg text-sm font-bold text-white"
              style={{ backgroundColor: awayColors.primary }}
            >
              {game.away_team}
            </div>
            <span className="text-text-muted font-medium">@</span>
            <div
              className="px-3 py-1 rounded-lg text-sm font-bold text-white"
              style={{ backgroundColor: homeColors.primary }}
            >
              {game.home_team}
            </div>
          </div>
          <Badge variant="info" size="sm">
            {valueBets.length} Value Bets
          </Badge>
        </div>

        {/* Model Prediction Summary */}
        {predictions?.spread && (
          <div className="bg-surface-secondary rounded-lg p-3">
            <div className="flex items-center justify-between text-sm">
              <span className="text-text-muted">Model Spread:</span>
              <span className="font-semibold text-brand-600 dark:text-brand-400">
                {predictions.spread.predicted_spread > 0 ? '+' : ''}
                {predictions.spread.predicted_spread.toFixed(1)}
              </span>
            </div>
            {game.vegas_line !== null && (
              <div className="flex items-center justify-between text-sm mt-1">
                <span className="text-text-muted">Vegas Line:</span>
                <span className="font-semibold text-text-primary">
                  {game.vegas_line > 0 ? '+' : ''}
                  {game.vegas_line.toFixed(1)}
                </span>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Bet Categories */}
      {(Object.entries(categoryConfig) as [BetCategory, typeof categoryConfig[BetCategory]][]).map(
        ([category, config]) => {
          const bets = categorizedBets[category];
          const isExpanded = expandedCategories.has(category);
          const Icon = config.icon;

          return (
            <div
              key={category}
              className="bg-surface-primary rounded-xl border border-surface-border overflow-hidden"
            >
              {/* Category Header */}
              <button
                onClick={() => toggleCategory(category)}
                className="w-full flex items-center justify-between p-4 hover:bg-surface-secondary/50 transition-colors"
              >
                <div className="flex items-center gap-3">
                  <div className={`p-2 rounded-lg bg-surface-secondary ${config.color}`}>
                    <Icon className="w-4 h-4" />
                  </div>
                  <span className="font-semibold text-text-primary">{config.label}</span>
                  <Badge variant={bets.length > 0 ? 'success' : 'default'} size="sm">
                    {bets.length}
                  </Badge>
                </div>
                <ChevronDown
                  className={`w-5 h-5 text-text-muted transition-transform duration-200 ${
                    isExpanded ? 'rotate-180' : ''
                  }`}
                />
              </button>

              {/* Category Content */}
              <AnimatePresence>
                {isExpanded && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: 'auto', opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    transition={{ duration: 0.2 }}
                    className="overflow-hidden"
                  >
                    <div className="px-4 pb-4 space-y-3">
                      {bets.length === 0 ? (
                        <div className="text-center py-6 text-text-muted">
                          <Target className="w-8 h-8 mx-auto mb-2 opacity-50" />
                          <p>No value bets found for {config.label.toLowerCase()}</p>
                        </div>
                      ) : (
                        bets.map((bet, index) => (
                          <BetCard key={`${bet.description}-${index}`} bet={bet} category={category} />
                        ))
                      )}

                      {/* Show prop predictions if available */}
                      {category === 'props' && predictions?.player_props && predictions.player_props.length > 0 && (
                        <div className="mt-4 pt-4 border-t border-surface-border">
                          <h4 className="text-sm font-medium text-text-secondary mb-3 flex items-center gap-2">
                            <User className="w-4 h-4" />
                            Model Predictions
                          </h4>
                          <div className="space-y-2">
                            {predictions.player_props.slice(0, 5).map((prop, index) => (
                              <PropPredictionCard key={`${prop.player_name}-${prop.prop_type}-${index}`} prop={prop} />
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          );
        }
      )}
    </motion.div>
  );
}

function BetCard({ bet, category }: { bet: ValueBet; category: BetCategory }) {
  const formatOdds = (odds: number) => (odds > 0 ? `+${odds}` : odds.toString());
  const formatPercent = (value: number) => `${(value * 100).toFixed(1)}%`;
  const formatCurrency = (value: number | null) => {
    if (value === null) return 'N/A';
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
    }).format(value);
  };

  const urgencyVariant = {
    CRITICAL: 'danger' as const,
    HIGH: 'warning' as const,
    MEDIUM: 'info' as const,
    LOW: 'default' as const,
  };

  return (
    <div className="bg-surface-secondary rounded-lg p-3 hover:bg-surface-elevated transition-colors">
      <div className="flex items-start justify-between mb-2">
        <div className="flex-1">
          <h4 className="font-medium text-text-primary">{bet.description}</h4>
          <p className="text-xs text-text-muted">{bet.bookmaker}</p>
        </div>
        <Badge variant={urgencyVariant[bet.urgency as keyof typeof urgencyVariant] || 'default'} size="sm">
          {bet.urgency}
        </Badge>
      </div>

      <div className="grid grid-cols-4 gap-2 text-sm">
        <div>
          <span className="text-text-muted text-xs flex items-center gap-1">
            <Calculator className="w-3 h-3" />
            Odds
          </span>
          <span className="font-semibold text-text-primary">{formatOdds(bet.odds)}</span>
        </div>
        <div>
          <span className="text-text-muted text-xs flex items-center gap-1">
            <TrendingUp className="w-3 h-3" />
            Edge
          </span>
          <span className="font-semibold text-success">{formatPercent(bet.edge)}</span>
        </div>
        <div>
          <span className="text-text-muted text-xs flex items-center gap-1">
            <Percent className="w-3 h-3" />
            EV
          </span>
          <span className="font-semibold text-success">{formatPercent(bet.expected_value)}</span>
        </div>
        <div>
          <span className="text-text-muted text-xs flex items-center gap-1">
            <DollarSign className="w-3 h-3" />
            Stake
          </span>
          <span className="font-semibold text-brand-600 dark:text-brand-400">
            {formatCurrency(bet.recommended_stake)}
          </span>
        </div>
      </div>
    </div>
  );
}

function PropPredictionCard({ prop }: { prop: PropPrediction }) {
  return (
    <div className="bg-surface-elevated/50 rounded-lg p-2.5 text-sm">
      <div className="flex items-center justify-between">
        <div>
          <span className="font-medium text-text-primary">{prop.player_name}</span>
          <span className="text-text-muted ml-2 text-xs">{prop.team}</span>
        </div>
        {prop.recommendation && (
          <Badge variant={prop.recommendation === 'OVER' ? 'success' : 'danger'} size="sm">
            {prop.recommendation}
          </Badge>
        )}
      </div>
      <div className="flex items-center gap-4 mt-1 text-xs">
        <span className="text-text-muted">
          {prop.prop_type.replace('_', ' ')}: <span className="font-medium text-text-primary">{prop.predicted_value.toFixed(1)}</span>
        </span>
        {prop.dk_line && (
          <span className="text-text-muted">
            Line: <span className="font-medium text-text-primary">{prop.dk_line}</span>
          </span>
        )}
        {prop.edge && (
          <span className="text-success">
            {(prop.edge * 100).toFixed(1)}% edge
          </span>
        )}
      </div>
    </div>
  );
}
