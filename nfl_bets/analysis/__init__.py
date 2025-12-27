"""
Comprehensive model analysis tools.

Provides:
- Backtesting with seasonal breakdown
- Calibration analysis and visualization
- Edge validation and threshold optimization
- Model vs Vegas comparison
- Feature importance and stability analysis
"""

from .backtest_analysis import (
    SeasonalBacktester,
    SeasonResult,
    SeasonalResults,
    EdgeDecayAnalysis,
)
from .calibration_analysis import (
    CalibrationAnalyzer,
    CalibrationReport,
    ReliabilityDiagram,
)
from .edge_validation import (
    EdgeValidator,
    EdgeThresholdResult,
    EdgeBucketAnalysis,
)
from .model_comparison import (
    ModelComparison,
    VegasBenchmark,
    ComparisonResult,
)
from .feature_analysis import (
    FeatureAnalyzer,
    FeatureStability,
    FeatureImportanceReport,
)

__all__ = [
    # Backtesting
    "SeasonalBacktester",
    "SeasonResult",
    "SeasonalResults",
    "EdgeDecayAnalysis",
    # Calibration
    "CalibrationAnalyzer",
    "CalibrationReport",
    "ReliabilityDiagram",
    # Edge validation
    "EdgeValidator",
    "EdgeThresholdResult",
    "EdgeBucketAnalysis",
    # Model comparison
    "ModelComparison",
    "VegasBenchmark",
    "ComparisonResult",
    # Feature analysis
    "FeatureAnalyzer",
    "FeatureStability",
    "FeatureImportanceReport",
]
