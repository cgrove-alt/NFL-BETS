'use client';

import { BankrollSummary } from '@/lib/api';

interface BankrollWidgetProps {
  bankroll: BankrollSummary | null;
  isLoading?: boolean;
}

export default function BankrollWidget({ bankroll, isLoading }: BankrollWidgetProps) {
  if (isLoading) {
    return (
      <div className="bg-white rounded-lg shadow-md p-4 animate-pulse">
        <div className="h-4 bg-gray-200 rounded w-1/3 mb-4"></div>
        <div className="h-8 bg-gray-200 rounded w-1/2 mb-2"></div>
        <div className="h-4 bg-gray-200 rounded w-2/3"></div>
      </div>
    );
  }

  if (!bankroll) {
    return (
      <div className="bg-white rounded-lg shadow-md p-4">
        <p className="text-gray-500">Unable to load bankroll data</p>
      </div>
    );
  }

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
    }).format(value);
  };

  const formatPercent = (value: number) => {
    const sign = value >= 0 ? '+' : '';
    return `${sign}${(value * 100).toFixed(1)}%`;
  };

  const profitColor = bankroll.total_profit >= 0 ? 'text-green-600' : 'text-red-600';

  return (
    <div className="bg-white rounded-lg shadow-md p-4">
      <h3 className="text-sm font-medium text-gray-500 mb-2">Bankroll</h3>

      <div className="text-3xl font-bold text-gray-900 mb-1">
        {formatCurrency(bankroll.current_bankroll)}
      </div>

      <div className={`text-sm font-medium ${profitColor}`}>
        {formatCurrency(bankroll.total_profit)} ({formatPercent(bankroll.roi)})
      </div>

      <div className="mt-4 grid grid-cols-2 gap-4">
        <div>
          <p className="text-xs text-gray-500">Available</p>
          <p className="font-medium">{formatCurrency(bankroll.available_bankroll)}</p>
        </div>
        <div>
          <p className="text-xs text-gray-500">Pending</p>
          <p className="font-medium text-yellow-600">
            {formatCurrency(bankroll.pending_exposure)}
          </p>
        </div>
      </div>
    </div>
  );
}
