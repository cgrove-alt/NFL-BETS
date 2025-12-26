'use client';

import { Brain, CheckCircle, AlertTriangle, XCircle, Calendar, Activity } from 'lucide-react';
import { ModelInfo } from '@/lib/api';
import { Badge } from './ui/Badge';

interface ModelStatusBadgeProps {
  model: ModelInfo;
}

export default function ModelStatusBadge({ model }: ModelStatusBadgeProps) {
  const formatModelName = (name: string) => {
    return name
      .split('_')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  };

  if (model.error) {
    return (
      <div className="bg-danger-light dark:bg-danger/10 rounded-lg p-3 border border-danger/20">
        <div className="flex justify-between items-center">
          <div className="flex items-center gap-2">
            <XCircle className="w-4 h-4 text-danger" />
            <span className="font-medium text-text-primary">{formatModelName(model.model_type)}</span>
          </div>
          <Badge variant="danger" size="sm">Error</Badge>
        </div>
        <p className="text-sm text-danger mt-1.5 pl-6">{model.error}</p>
      </div>
    );
  }

  const StatusIcon = model.is_stale ? AlertTriangle : CheckCircle;
  const statusVariant = model.is_stale ? 'warning' : 'success';

  return (
    <div className="bg-surface-secondary rounded-lg p-3 border border-surface-border hover:border-brand-300 dark:hover:border-brand-700 transition-colors">
      <div className="flex justify-between items-center">
        <div className="flex items-center gap-2">
          <Brain className="w-4 h-4 text-brand-500" />
          <span className="font-medium text-text-primary">{formatModelName(model.model_type)}</span>
        </div>
        <Badge variant={statusVariant} size="sm">
          <StatusIcon className="w-3 h-3 mr-1" />
          {model.is_stale ? 'Stale' : 'Fresh'}
        </Badge>
      </div>

      <div className="mt-2 pl-6 space-y-1">
        {model.data_cutoff_date && (
          <p className="flex items-center gap-1.5 text-xs text-text-muted">
            <Calendar className="w-3 h-3" />
            Data: {new Date(model.data_cutoff_date).toLocaleDateString()}
          </p>
        )}

        {model.metrics && (
          <div className="flex items-center gap-3 text-xs text-text-secondary">
            <Activity className="w-3 h-3 text-text-muted" />
            {model.metrics.mae !== undefined && (
              <span className="bg-surface-primary px-1.5 py-0.5 rounded">
                MAE: <span className="font-medium">{model.metrics.mae.toFixed(2)}</span>
              </span>
            )}
            {model.metrics.r2 !== undefined && (
              <span className="bg-surface-primary px-1.5 py-0.5 rounded">
                R²: <span className="font-medium">{model.metrics.r2.toFixed(3)}</span>
              </span>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
