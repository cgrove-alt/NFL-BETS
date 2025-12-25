'use client';

import { useState } from 'react';
import { ConfidenceGauge } from './ConfidenceGauge';
import { GameInfo, ValueBet } from '@/lib/api';

interface GameDetailModalProps {
  game: GameInfo;
  valueBets: ValueBet[];
  onClose: () => void;
}

type BetFilter = 'all' | 'spread' | 'moneyline' | 'total' | 'props';

export function GameDetailModal({ game, valueBets, onClose }: GameDetailModalProps) {
  const [activeFilter, setActiveFilter] = useState<BetFilter>('all');

  const kickoffDate = new Date(game.kickoff);

  const filterBets = (bets: ValueBet[], filter: BetFilter) => {
    if (filter === 'all') return bets;
    if (filter === 'spread') return bets.filter(b => b.bet_type === 'spread');
    if (filter === 'moneyline') return bets.filter(b => b.bet_type === 'moneyline' || b.bet_type === 'h2h');
    if (filter === 'total') return bets.filter(b => b.bet_type === 'total' || b.bet_type === 'over_under');
    if (filter === 'props') return bets.filter(b =>
      b.bet_type.includes('yards') ||
      b.bet_type.includes('td') ||
      b.bet_type.includes('receptions')
    );
    return bets;
  };

  const filteredBets = filterBets(valueBets, activeFilter);

  const getUrgencyStyles = (urgency: string) => {
    switch (urgency.toUpperCase()) {
      case 'CRITICAL':
        return { bg: 'bg-red-100', text: 'text-red-700', border: 'border-red-300', icon: '⚡' };
      case 'HIGH':
        return { bg: 'bg-orange-100', text: 'text-orange-700', border: 'border-orange-300', icon: '🔥' };
      case 'MEDIUM':
        return { bg: 'bg-yellow-100', text: 'text-yellow-700', border: 'border-yellow-300', icon: '⚠️' };
      default:
        return { bg: 'bg-gray-100', text: 'text-gray-700', border: 'border-gray-300', icon: '📊' };
    }
  };

  const formatOdds = (odds: number) => {
    return odds > 0 ? `+${odds}` : odds.toString();
  };

  const getTimeUntilKickoff = () => {
    const now = new Date();
    const diff = kickoffDate.getTime() - now.getTime();
    if (diff <= 0) return 'Game Started';

    const hours = Math.floor(diff / (1000 * 60 * 60));
    const minutes = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60));

    if (hours > 24) {
      const days = Math.floor(hours / 24);
      return `${days}d ${hours % 24}h`;
    }
    return `${hours}h ${minutes}m`;
  };

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-2xl max-w-3xl w-full max-h-[90vh] overflow-hidden shadow-2xl">
        {/* Header */}
        <div className="bg-gradient-to-r from-blue-600 to-blue-800 text-white p-6">
          <div className="flex justify-between items-start">
            <button
              onClick={onClose}
              className="text-white/80 hover:text-white transition-colors"
            >
              ← Back
            </button>
            <span className="text-sm bg-white/20 px-3 py-1 rounded-full">
              Week {game.week}
            </span>
          </div>

          <div className="text-center mt-4">
            <div className="text-3xl font-bold mb-2">
              {game.away_team} <span className="text-white/60">@</span> {game.home_team}
            </div>
            <div className="text-blue-100">
              {kickoffDate.toLocaleDateString('en-US', {
                weekday: 'long',
                month: 'long',
                day: 'numeric'
              })} • {kickoffDate.toLocaleTimeString('en-US', {
                hour: 'numeric',
                minute: '2-digit',
                hour12: true
              })}
            </div>
            <div className="text-sm text-blue-200 mt-1">
              Kickoff in {getTimeUntilKickoff()}
            </div>
          </div>

          {/* Model vs Vegas Comparison */}
          {game.model_prediction !== null && game.vegas_line !== null && (
            <div className="flex justify-center gap-8 mt-6">
              <div className="text-center">
                <div className="text-sm text-blue-200">Model Pick</div>
                <div className="text-2xl font-bold">
                  {game.away_team} {game.model_prediction > 0 ? '+' : ''}{game.model_prediction?.toFixed(1)}
                </div>
              </div>
              <div className="text-white/40 text-2xl">vs</div>
              <div className="text-center">
                <div className="text-sm text-blue-200">Vegas Line</div>
                <div className="text-2xl font-bold">
                  {game.away_team} {game.vegas_line > 0 ? '+' : ''}{game.vegas_line?.toFixed(1)}
                </div>
              </div>
            </div>
          )}

          {/* Confidence Gauge */}
          {game.model_confidence && (
            <div className="mt-6 max-w-md mx-auto">
              <ConfidenceGauge
                value={game.model_confidence}
                size="lg"
                label="Model Confidence"
              />
            </div>
          )}
        </div>

        {/* Content */}
        <div className="p-6 overflow-y-auto max-h-[50vh]">
          {/* Filter Tabs */}
          <div className="flex gap-2 mb-6 overflow-x-auto pb-2">
            {(['all', 'spread', 'moneyline', 'total', 'props'] as BetFilter[]).map(filter => (
              <button
                key={filter}
                onClick={() => setActiveFilter(filter)}
                className={`
                  px-4 py-2 rounded-full text-sm font-medium whitespace-nowrap transition-colors
                  ${activeFilter === filter
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                  }
                `}
              >
                {filter.charAt(0).toUpperCase() + filter.slice(1)}
                {filter === 'all' && ` (${valueBets.length})`}
              </button>
            ))}
          </div>

          {/* Value Bets List */}
          {filteredBets.length === 0 ? (
            <div className="text-center py-12 text-gray-500">
              <div className="text-4xl mb-2">🎯</div>
              <div>No value bets found for this filter</div>
            </div>
          ) : (
            <div className="space-y-4">
              {filteredBets.map((bet, index) => {
                const styles = getUrgencyStyles(bet.urgency);
                return (
                  <div
                    key={`${bet.game_id}-${bet.bet_type}-${index}`}
                    className={`
                      ${styles.bg} border ${styles.border} rounded-xl p-4
                      transition-all hover:shadow-md
                    `}
                  >
                    {/* Header */}
                    <div className="flex justify-between items-start mb-3">
                      <div>
                        <span className={`text-xs font-bold ${styles.text} uppercase`}>
                          {styles.icon} {bet.urgency}
                        </span>
                        <h3 className="text-lg font-bold text-gray-800 mt-1">
                          {bet.description}
                        </h3>
                      </div>
                      <span className="text-sm text-gray-500 bg-white px-2 py-1 rounded">
                        {bet.bookmaker}
                      </span>
                    </div>

                    {/* Metrics Grid */}
                    <div className="grid grid-cols-3 gap-4 mb-4">
                      <div className="bg-white/50 rounded-lg p-3 text-center">
                        <div className="text-xs text-gray-500 mb-1">Edge</div>
                        <div className="text-xl font-bold text-green-600">
                          +{(bet.edge * 100).toFixed(1)}%
                        </div>
                      </div>
                      <div className="bg-white/50 rounded-lg p-3 text-center">
                        <div className="text-xs text-gray-500 mb-1">Model Prob</div>
                        <div className="text-xl font-bold text-blue-600">
                          {(bet.model_probability * 100).toFixed(1)}%
                        </div>
                      </div>
                      <div className="bg-white/50 rounded-lg p-3 text-center">
                        <div className="text-xs text-gray-500 mb-1">Odds</div>
                        <div className="text-xl font-bold text-gray-800">
                          {formatOdds(bet.odds)}
                        </div>
                      </div>
                    </div>

                    {/* Confidence Bar */}
                    <div className="mb-4">
                      <ConfidenceGauge
                        value={bet.model_probability}
                        size="sm"
                        label="Prediction Confidence"
                      />
                    </div>

                    {/* Stake Recommendation */}
                    <div className="flex justify-between items-center bg-white/50 rounded-lg p-3">
                      <div>
                        <span className="text-sm text-gray-600">Recommended Stake</span>
                        <div className="text-lg font-bold text-blue-600">
                          {bet.recommended_stake
                            ? `$${bet.recommended_stake.toFixed(0)}`
                            : 'Calculate based on bankroll'
                          }
                        </div>
                      </div>
                      <div className="text-right">
                        <span className="text-sm text-gray-600">Expected Value</span>
                        <div className="text-lg font-bold text-green-600">
                          +{(bet.expected_value * 100).toFixed(1)}%
                        </div>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
