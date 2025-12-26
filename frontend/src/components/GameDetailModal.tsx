'use client';

import { useState, useEffect } from 'react';
import { ConfidenceGauge } from './ConfidenceGauge';
import { GameInfo, ValueBet, PropPrediction, getGamePredictions } from '@/lib/api';

interface GameDetailModalProps {
  game: GameInfo;
  valueBets: ValueBet[];
  onClose: () => void;
}

type BetFilter = 'all' | 'spread' | 'moneyline' | 'total' | 'props';

export function GameDetailModal({ game, valueBets, onClose }: GameDetailModalProps) {
  const [activeFilter, setActiveFilter] = useState<BetFilter>('all');
  const [propPredictions, setPropPredictions] = useState<PropPrediction[]>([]);
  const [loadingProps, setLoadingProps] = useState(false);

  // Fetch prop predictions when props tab is selected
  useEffect(() => {
    if (activeFilter === 'props' && propPredictions.length === 0) {
      setLoadingProps(true);
      getGamePredictions(game.game_id)
        .then((result) => {
          if (result && result.player_props) {
            setPropPredictions(result.player_props);
          }
        })
        .catch((err) => {
          console.error('Failed to fetch prop predictions:', err);
        })
        .finally(() => {
          setLoadingProps(false);
        });
    }
  }, [activeFilter, game.game_id, propPredictions.length]);

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

  const formatPropType = (propType: string) => {
    const typeMap: Record<string, string> = {
      'passing_yards': 'Pass Yds',
      'rushing_yards': 'Rush Yds',
      'receiving_yards': 'Rec Yds',
      'receptions': 'Receptions',
    };
    return typeMap[propType] || propType;
  };

  const getInjuryBadge = (status: string | undefined) => {
    if (!status || status === 'ACTIVE' || status === 'UNKNOWN') return null;
    const colors: Record<string, string> = {
      'QUESTIONABLE': 'bg-yellow-100 text-yellow-700',
      'DOUBTFUL': 'bg-orange-100 text-orange-700',
      'OUT': 'bg-red-100 text-red-700',
      'IR': 'bg-red-100 text-red-700',
    };
    return (
      <span className={`text-xs px-2 py-0.5 rounded ${colors[status] || 'bg-gray-100 text-gray-600'}`}>
        {status}
      </span>
    );
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

          {/* Props Tab - Show Predictions */}
          {activeFilter === 'props' && (
            <>
              {loadingProps ? (
                <div className="text-center py-12 text-gray-500">
                  <div className="animate-spin text-4xl mb-2">⚙️</div>
                  <div>Loading player predictions...</div>
                </div>
              ) : propPredictions.length > 0 ? (
                <div className="space-y-4">
                  {/* Group by team */}
                  {[game.home_team, game.away_team].map(team => {
                    const teamPreds = propPredictions.filter(p => p.team === team);
                    if (teamPreds.length === 0) return null;
                    return (
                      <div key={team}>
                        <h3 className="text-sm font-bold text-gray-500 mb-2 uppercase">{team}</h3>
                        <div className="grid gap-3">
                          {teamPreds.map((pred, idx) => (
                            <div
                              key={`${pred.player_name}-${pred.prop_type}-${idx}`}
                              className="bg-gradient-to-r from-purple-50 to-blue-50 border border-purple-200 rounded-xl p-4"
                            >
                              <div className="flex justify-between items-start mb-2">
                                <div>
                                  <div className="flex items-center gap-2">
                                    <span className="font-bold text-gray-800">{pred.player_name}</span>
                                    {getInjuryBadge(pred.injury_status)}
                                  </div>
                                  <span className="text-sm text-purple-600 font-medium">
                                    {formatPropType(pred.prop_type)}
                                  </span>
                                </div>
                                <div className="text-right">
                                  <div className="text-2xl font-bold text-purple-700">
                                    {pred.predicted_value.toFixed(0)}
                                  </div>
                                  <div className="text-xs text-gray-500">
                                    Range: {pred.range_low.toFixed(0)} - {pred.range_high.toFixed(0)}
                                  </div>
                                </div>
                              </div>
                              <div className="mt-2">
                                <ConfidenceGauge
                                  value={pred.confidence}
                                  size="sm"
                                  label="Confidence"
                                />
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    );
                  })}

                  {/* Show value bets if any */}
                  {filteredBets.length > 0 && (
                    <div className="mt-6">
                      <h3 className="text-sm font-bold text-green-600 mb-2 uppercase">Value Bets Found</h3>
                      <div className="space-y-3">
                        {filteredBets.map((bet, index) => {
                          const styles = getUrgencyStyles(bet.urgency);
                          return (
                            <div
                              key={`${bet.game_id}-${bet.bet_type}-${index}`}
                              className={`${styles.bg} border ${styles.border} rounded-xl p-4`}
                            >
                              <div className="flex justify-between items-start">
                                <div>
                                  <span className={`text-xs font-bold ${styles.text} uppercase`}>
                                    {styles.icon} {bet.urgency}
                                  </span>
                                  <h4 className="font-bold text-gray-800">{bet.description}</h4>
                                </div>
                                <div className="text-right">
                                  <div className="text-lg font-bold text-green-600">+{(bet.edge * 100).toFixed(1)}%</div>
                                  <div className="text-sm text-gray-500">{bet.bookmaker}</div>
                                </div>
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <div className="text-center py-12 text-gray-500">
                  <div className="text-4xl mb-2">🏈</div>
                  <div>No player predictions available yet</div>
                  <div className="text-sm mt-1">Check back closer to game time</div>
                </div>
              )}
            </>
          )}

          {/* Non-Props Tabs - Show Value Bets */}
          {activeFilter !== 'props' && (
            <>
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
            </>
          )}
        </div>
      </div>
    </div>
  );
}
