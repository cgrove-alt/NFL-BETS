'use client';

import { ValueBet } from '@/lib/api';

interface ValueBetCardProps {
  bet: ValueBet;
}

const urgencyColors: Record<string, string> = {
  CRITICAL: 'bg-red-500 text-white',
  HIGH: 'bg-orange-500 text-white',
  MEDIUM: 'bg-yellow-500 text-black',
  LOW: 'bg-gray-300 text-black',
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
    return `$${value.toFixed(0)}`;
  };

  return (
    <div className="bg-white rounded-lg shadow-md p-4 border-l-4 border-green-500">
      <div className="flex justify-between items-start mb-2">
        <div>
          <h3 className="font-semibold text-lg">{bet.description}</h3>
          <p className="text-sm text-gray-500">{bet.bookmaker}</p>
        </div>
        <span
          className={`px-2 py-1 rounded text-xs font-medium ${urgencyColors[bet.urgency] || urgencyColors.LOW}`}
        >
          {bet.urgency}
        </span>
      </div>

      <div className="grid grid-cols-3 gap-4 mt-3">
        <div>
          <p className="text-xs text-gray-500">Odds</p>
          <p className="font-medium">{formatOdds(bet.odds)}</p>
        </div>
        <div>
          <p className="text-xs text-gray-500">Edge</p>
          <p className="font-medium text-green-600">{formatPercent(bet.edge)}</p>
        </div>
        <div>
          <p className="text-xs text-gray-500">EV</p>
          <p className="font-medium text-green-600">{formatPercent(bet.expected_value)}</p>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4 mt-3">
        <div>
          <p className="text-xs text-gray-500">Model Prob</p>
          <p className="font-medium">{formatPercent(bet.model_probability)}</p>
        </div>
        <div>
          <p className="text-xs text-gray-500">Rec. Stake</p>
          <p className="font-medium text-blue-600">{formatCurrency(bet.recommended_stake)}</p>
        </div>
      </div>

      <div className="mt-3 pt-3 border-t border-gray-100">
        <p className="text-xs text-gray-400">
          Detected: {new Date(bet.detected_at).toLocaleString()}
        </p>
      </div>
    </div>
  );
}
