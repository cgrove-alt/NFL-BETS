"""
Machine learning models for NFL betting predictions.

This module provides:
- Spread prediction model (XGBoost ensemble)
- Player prop models (quantile regression)
- Probability calibration
- Walk-forward validation

Example:
    >>> from nfl_bets.models import SpreadModel, WalkForwardValidator
    >>>
    >>> # Train spread model
    >>> model = SpreadModel()
    >>> model.train(X_train, y_train)
    >>>
    >>> # Validate with walk-forward
    >>> validator = WalkForwardValidator(train_seasons=3)
    >>> results = validator.validate(model, df, "actual_spread", feature_cols)
    >>> print(results.summary())
    >>>
    >>> # Make predictions
    >>> prediction = model.predict_game(
    ...     features, game_id="2024_01_KC_BAL",
    ...     home_team="BAL", away_team="KC", line=-3.5
    ... )
"""

from .base import (
    BaseModel,
    EnsembleModel,
    ModelMetadata,
    ModelMetrics,
    PredictionResult,
    QuantileModel,
)
from .calibration import (
    CalibrationMetrics,
    ProbabilityCalibrator,
    TemperatureScaling,
    calibrate_spread_probabilities,
    reliability_diagram_data,
)
from .evaluation import (
    FoldResult,
    TimeSeriesSplit,
    ValidationResults,
    WalkForwardValidator,
    brier_score,
    calculate_calibration_curve,
    evaluate_against_closing_line,
    expected_calibration_error,
    log_loss,
)
from .spread_model import SpreadModel, SpreadPrediction
from .residual_spread_model import ResidualSpreadModel, ResidualPrediction, ResidualModelConfig
from .moneyline_model import MoneylineModel, MoneylinePrediction
from .totals_model import TotalsModel, TotalsPrediction
from .player_props import (
    BasePropModel,
    PassingYardsModel,
    PropPrediction,
    ReceivingYardsModel,
    ReceptionsModel,
    RushingYardsModel,
)
from .model_manager import (
    ModelManager,
    get_model_manager_with_latest_date,
)

__all__ = [
    # Base classes
    "BaseModel",
    "EnsembleModel",
    "QuantileModel",
    "ModelMetadata",
    "ModelMetrics",
    "PredictionResult",
    # Calibration
    "ProbabilityCalibrator",
    "CalibrationMetrics",
    "TemperatureScaling",
    "calibrate_spread_probabilities",
    "reliability_diagram_data",
    # Evaluation
    "WalkForwardValidator",
    "ValidationResults",
    "FoldResult",
    "TimeSeriesSplit",
    "expected_calibration_error",
    "brier_score",
    "log_loss",
    "calculate_calibration_curve",
    "evaluate_against_closing_line",
    # Spread model
    "SpreadModel",
    "SpreadPrediction",
    # Residual spread model
    "ResidualSpreadModel",
    "ResidualPrediction",
    "ResidualModelConfig",
    # Moneyline model
    "MoneylineModel",
    "MoneylinePrediction",
    # Totals model
    "TotalsModel",
    "TotalsPrediction",
    # Prop models
    "BasePropModel",
    "PropPrediction",
    "PassingYardsModel",
    "RushingYardsModel",
    "ReceivingYardsModel",
    "ReceptionsModel",
    # Model management
    "ModelManager",
    "get_model_manager_with_latest_date",
]
