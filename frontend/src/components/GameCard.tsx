'use client';

import { ConfidenceCircle } from './ConfidenceGauge';
import { GameInfo } from '@/lib/api';

interface GameCardProps {
  game: GameInfo;
  onClick: () => void;
  isSelected?: boolean;
}

// NFL team colors (abbreviated)
const teamColors: Record<string, { primary: string; secondary: string }> = {
  KC: { primary: '#E31837', secondary: '#FFB81C' },
  BUF: { primary: '#00338D', secondary: '#C60C30' },
  SF: { primary: '#AA0000', secondary: '#B3995D' },
  DET: { primary: '#0076B6', secondary: '#B0B7BC' },
  PHI: { primary: '#004C54', secondary: '#A5ACAF' },
  DAL: { primary: '#003594', secondary: '#869397' },
  BAL: { primary: '#241773', secondary: '#9E7C0C' },
  MIA: { primary: '#008E97', secondary: '#FC4C02' },
  GB: { primary: '#203731', secondary: '#FFB612' },
  MIN: { primary: '#4F2683', secondary: '#FFC62F' },
  // Add more teams as needed
};

const getTeamColor = (team: string) => {
  return teamColors[team]?.primary || '#6B7280';
};

export function GameCard({ game, onClick, isSelected }: GameCardProps) {
  const kickoffDate = new Date(game.kickoff);
  const isToday = new Date().toDateString() === kickoffDate.toDateString();

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString('en-US', {
      hour: 'numeric',
      minute: '2-digit',
      hour12: true
    });
  };

  const formatDate = (date: Date) => {
    if (isToday) return 'Today';
    return date.toLocaleDateString('en-US', {
      weekday: 'short',
      month: 'short',
      day: 'numeric'
    });
  };

  const getUrgencyColor = (count: number, edge: number | null) => {
    if (count === 0) return 'bg-gray-200';
    if (edge && edge >= 0.05) return 'bg-red-500';
    if (edge && edge >= 0.03) return 'bg-orange-500';
    return 'bg-yellow-500';
  };

  const edgePercentage = game.best_edge ? `${(game.best_edge * 100).toFixed(1)}%` : null;

  return (
    <div
      onClick={onClick}
      className={`
        bg-white rounded-xl shadow-sm border-2 p-4 cursor-pointer
        transition-all duration-200 hover:shadow-md hover:border-blue-300
        ${isSelected ? 'border-blue-500 ring-2 ring-blue-200' : 'border-gray-100'}
      `}
    >
      {/* Teams and Time */}
      <div className="flex justify-between items-start mb-3">
        <div className="flex-1">
          <div className="flex items-center gap-2 mb-1">
            <div
              className="w-3 h-3 rounded-full"
              style={{ backgroundColor: getTeamColor(game.away_team) }}
            />
            <span className="font-bold text-gray-800">{game.away_team}</span>
            <span className="text-gray-400">@</span>
            <div
              className="w-3 h-3 rounded-full"
              style={{ backgroundColor: getTeamColor(game.home_team) }}
            />
            <span className="font-bold text-gray-800">{game.home_team}</span>
          </div>
          <div className="text-sm text-gray-500">
            {formatDate(kickoffDate)} • {formatTime(kickoffDate)}
          </div>
        </div>

        {/* Confidence Circle */}
        {game.model_confidence && game.value_bet_count > 0 && (
          <ConfidenceCircle value={game.model_confidence} size={50} />
        )}
      </div>

      {/* Value Bet Indicator Bar */}
      <div className="mb-3">
        <div className="flex justify-between text-xs text-gray-500 mb-1">
          <span>Value Bets</span>
          <span className="font-medium">{game.value_bet_count}</span>
        </div>
        <div className="w-full bg-gray-100 rounded-full h-2 overflow-hidden">
          <div
            className={`h-2 rounded-full transition-all duration-300 ${getUrgencyColor(game.value_bet_count, game.best_edge)}`}
            style={{ width: `${Math.min(game.value_bet_count * 20, 100)}%` }}
          />
        </div>
      </div>

      {/* Best Edge Preview */}
      {game.value_bet_count > 0 && game.best_bet_description ? (
        <div className="bg-green-50 rounded-lg p-2 border border-green-100">
          <div className="flex justify-between items-center">
            <span className="text-sm text-gray-700">{game.best_bet_description}</span>
            <span className="text-sm font-bold text-green-600">{edgePercentage} edge</span>
          </div>
        </div>
      ) : (
        <div className="bg-gray-50 rounded-lg p-2 border border-gray-100">
          <span className="text-sm text-gray-400">No value bets found</span>
        </div>
      )}

      {/* Model vs Vegas (if available) */}
      {game.model_prediction !== null && game.vegas_line !== null && (
        <div className="mt-3 pt-3 border-t border-gray-100 flex justify-between text-xs">
          <div>
            <span className="text-gray-500">Model: </span>
            <span className="font-medium text-blue-600">
              {game.model_prediction > 0 ? '+' : ''}{game.model_prediction.toFixed(1)}
            </span>
          </div>
          <div>
            <span className="text-gray-500">Vegas: </span>
            <span className="font-medium text-gray-700">
              {game.vegas_line > 0 ? '+' : ''}{game.vegas_line.toFixed(1)}
            </span>
          </div>
        </div>
      )}
    </div>
  );
}

// Loading skeleton
export function GameCardSkeleton() {
  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-4 animate-pulse">
      <div className="flex justify-between items-start mb-3">
        <div className="flex-1">
          <div className="h-5 bg-gray-200 rounded w-32 mb-2" />
          <div className="h-4 bg-gray-200 rounded w-24" />
        </div>
        <div className="w-12 h-12 bg-gray-200 rounded-full" />
      </div>
      <div className="h-2 bg-gray-200 rounded-full mb-3" />
      <div className="h-10 bg-gray-100 rounded-lg" />
    </div>
  );
}
