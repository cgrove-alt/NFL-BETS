'use client';

import { ModelInfo } from '@/lib/api';

interface ModelStatusBadgeProps {
  model: ModelInfo;
}

export default function ModelStatusBadge({ model }: ModelStatusBadgeProps) {
  const statusColor = model.is_stale ? 'bg-yellow-100 text-yellow-800' : 'bg-green-100 text-green-800';
  const statusText = model.is_stale ? 'Stale' : 'Fresh';

  if (model.error) {
    return (
      <div className="bg-red-50 rounded-lg p-3 border border-red-200">
        <div className="flex justify-between items-center">
          <span className="font-medium capitalize">{model.model_type.replace('_', ' ')}</span>
          <span className="px-2 py-1 rounded text-xs bg-red-100 text-red-800">Error</span>
        </div>
        <p className="text-sm text-red-600 mt-1">{model.error}</p>
      </div>
    );
  }

  return (
    <div className="bg-gray-50 rounded-lg p-3 border border-gray-200">
      <div className="flex justify-between items-center">
        <span className="font-medium capitalize">{model.model_type.replace('_', ' ')}</span>
        <span className={`px-2 py-1 rounded text-xs ${statusColor}`}>{statusText}</span>
      </div>

      {model.data_cutoff_date && (
        <p className="text-xs text-gray-500 mt-1">
          Data: {new Date(model.data_cutoff_date).toLocaleDateString()}
        </p>
      )}

      {model.metrics && (
        <div className="flex gap-3 mt-2 text-xs">
          {model.metrics.mae !== undefined && (
            <span>MAE: {model.metrics.mae.toFixed(2)}</span>
          )}
          {model.metrics.r2 !== undefined && (
            <span>R²: {model.metrics.r2.toFixed(3)}</span>
          )}
        </div>
      )}
    </div>
  );
}
