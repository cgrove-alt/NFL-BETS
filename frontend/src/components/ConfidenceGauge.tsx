'use client';

interface ConfidenceGaugeProps {
  value: number; // 0-1 probability
  size?: 'sm' | 'md' | 'lg';
  showLabel?: boolean;
  label?: string;
}

export function ConfidenceGauge({
  value,
  size = 'md',
  showLabel = true,
  label = 'Model Confidence'
}: ConfidenceGaugeProps) {
  const percentage = Math.round(value * 100);

  // Determine confidence level and color
  const getConfidenceLevel = (pct: number) => {
    if (pct >= 80) return { level: 'Very High', color: 'bg-emerald-500', textColor: 'text-emerald-600' };
    if (pct >= 65) return { level: 'High', color: 'bg-green-500', textColor: 'text-green-600' };
    if (pct >= 55) return { level: 'Moderate', color: 'bg-yellow-500', textColor: 'text-yellow-600' };
    if (pct >= 50) return { level: 'Low', color: 'bg-orange-500', textColor: 'text-orange-600' };
    return { level: 'Very Low', color: 'bg-red-500', textColor: 'text-red-600' };
  };

  const { level, color, textColor } = getConfidenceLevel(percentage);

  const sizeClasses = {
    sm: { bar: 'h-2', text: 'text-xs', container: 'gap-1' },
    md: { bar: 'h-3', text: 'text-sm', container: 'gap-2' },
    lg: { bar: 'h-4', text: 'text-base', container: 'gap-2' }
  };

  const sizes = sizeClasses[size];

  return (
    <div className={`flex flex-col ${sizes.container}`}>
      {showLabel && (
        <div className="flex justify-between items-center">
          <span className={`${sizes.text} text-gray-600 font-medium`}>{label}</span>
          <span className={`${sizes.text} font-bold ${textColor}`}>{percentage}%</span>
        </div>
      )}
      <div className={`w-full bg-gray-200 rounded-full ${sizes.bar} overflow-hidden`}>
        <div
          className={`${sizes.bar} ${color} rounded-full transition-all duration-500 ease-out`}
          style={{ width: `${percentage}%` }}
        />
      </div>
      {showLabel && (
        <span className={`${sizes.text} ${textColor} font-medium`}>{level}</span>
      )}
    </div>
  );
}

// Circular gauge variant for compact display
export function ConfidenceCircle({
  value,
  size = 60
}: {
  value: number;
  size?: number;
}) {
  const percentage = Math.round(value * 100);
  const strokeWidth = size / 10;
  const radius = (size - strokeWidth) / 2;
  const circumference = radius * 2 * Math.PI;
  const offset = circumference - (value * circumference);

  const getColor = (pct: number) => {
    if (pct >= 80) return '#10b981'; // emerald-500
    if (pct >= 65) return '#22c55e'; // green-500
    if (pct >= 55) return '#eab308'; // yellow-500
    if (pct >= 50) return '#f97316'; // orange-500
    return '#ef4444'; // red-500
  };

  const color = getColor(percentage);

  return (
    <div className="relative inline-flex items-center justify-center">
      <svg width={size} height={size} className="transform -rotate-90">
        {/* Background circle */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="#e5e7eb"
          strokeWidth={strokeWidth}
        />
        {/* Progress circle */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke={color}
          strokeWidth={strokeWidth}
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          className="transition-all duration-500 ease-out"
        />
      </svg>
      <span
        className="absolute font-bold text-gray-700"
        style={{ fontSize: size / 4 }}
      >
        {percentage}%
      </span>
    </div>
  );
}
