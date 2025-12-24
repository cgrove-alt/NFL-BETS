"""
Walk-forward validation for NFL prediction models.

Implements proper time-series cross-validation to prevent look-ahead bias:
- Train on seasons N through N+k, test on season N+k+1
- Never use future data in training
- Track performance across multiple seasons
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional, Union

import numpy as np
import polars as pl
from loguru import logger

from .base import BaseModel, ModelMetrics


@dataclass
class FoldResult:
    """Results from a single validation fold."""

    fold_id: int
    train_seasons: list[int]
    test_season: int
    metrics: ModelMetrics
    predictions: np.ndarray
    actuals: np.ndarray
    lines: Optional[np.ndarray] = None
    feature_importance: dict[str, float] = field(default_factory=dict)


@dataclass
class ValidationResults:
    """Complete validation results across all folds."""

    fold_results: list[FoldResult]
    model_type: str
    validation_date: datetime = field(default_factory=datetime.now)

    @property
    def n_folds(self) -> int:
        """Number of validation folds."""
        return len(self.fold_results)

    @property
    def mean_mae(self) -> float:
        """Mean MAE across folds."""
        return np.mean([f.metrics.mae for f in self.fold_results])

    @property
    def std_mae(self) -> float:
        """Standard deviation of MAE across folds."""
        return np.std([f.metrics.mae for f in self.fold_results])

    @property
    def mean_rmse(self) -> float:
        """Mean RMSE across folds."""
        return np.mean([f.metrics.rmse for f in self.fold_results])

    @property
    def mean_r2(self) -> float:
        """Mean R² across folds."""
        return np.mean([f.metrics.r2 for f in self.fold_results])

    @property
    def total_ats_record(self) -> tuple[int, int, int]:
        """Combined ATS record across all folds."""
        wins = sum(f.metrics.ats_wins for f in self.fold_results)
        losses = sum(f.metrics.ats_losses for f in self.fold_results)
        pushes = sum(f.metrics.ats_pushes for f in self.fold_results)
        return (wins, losses, pushes)

    @property
    def ats_win_rate(self) -> float:
        """Overall ATS win rate."""
        wins, losses, _ = self.total_ats_record
        total = wins + losses
        return wins / total if total > 0 else 0.0

    @property
    def ats_roi(self) -> float:
        """Overall ATS ROI assuming -110 juice."""
        wins, losses, _ = self.total_ats_record
        total = wins + losses
        if total == 0:
            return 0.0
        profit = wins * 100 - losses * 110
        return profit / (total * 110)

    @property
    def n_total_samples(self) -> int:
        """Total samples across all folds."""
        return sum(f.metrics.n_samples for f in self.fold_results)

    def get_aggregate_feature_importance(self) -> dict[str, float]:
        """Get averaged feature importance across folds."""
        if not self.fold_results:
            return {}

        all_features: set[str] = set()
        for fold in self.fold_results:
            all_features.update(fold.feature_importance.keys())

        importance = {}
        for feature in all_features:
            values = [
                fold.feature_importance.get(feature, 0.0)
                for fold in self.fold_results
            ]
            importance[feature] = np.mean(values)

        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "n_folds": self.n_folds,
            "model_type": self.model_type,
            "validation_date": self.validation_date.isoformat(),
            "mean_mae": self.mean_mae,
            "std_mae": self.std_mae,
            "mean_rmse": self.mean_rmse,
            "mean_r2": self.mean_r2,
            "ats_record": self.total_ats_record,
            "ats_win_rate": self.ats_win_rate,
            "ats_roi": self.ats_roi,
            "n_total_samples": self.n_total_samples,
            "fold_details": [
                {
                    "fold_id": f.fold_id,
                    "train_seasons": f.train_seasons,
                    "test_season": f.test_season,
                    "metrics": f.metrics.to_dict(),
                }
                for f in self.fold_results
            ],
        }

    def summary(self) -> str:
        """Generate a summary string of validation results."""
        wins, losses, pushes = self.total_ats_record
        lines = [
            f"Walk-Forward Validation Results ({self.model_type})",
            "=" * 50,
            f"Folds: {self.n_folds}",
            f"Total Samples: {self.n_total_samples}",
            "",
            "Regression Metrics:",
            f"  MAE: {self.mean_mae:.3f} ± {self.std_mae:.3f}",
            f"  RMSE: {self.mean_rmse:.3f}",
            f"  R²: {self.mean_r2:.3f}",
            "",
            "Betting Metrics:",
            f"  ATS Record: {wins}-{losses}-{pushes}",
            f"  ATS Win Rate: {self.ats_win_rate:.1%}",
            f"  ATS ROI: {self.ats_roi:+.1%}",
            "",
            "Per-Fold Results:",
        ]

        for fold in self.fold_results:
            w, l, p = fold.metrics.ats_record
            lines.append(
                f"  Fold {fold.fold_id}: Train {fold.train_seasons} → "
                f"Test {fold.test_season} | "
                f"MAE: {fold.metrics.mae:.3f} | "
                f"ATS: {w}-{l}-{p}"
            )

        return "\n".join(lines)


class WalkForwardValidator:
    """
    Walk-forward cross-validation for time-series prediction models.

    Implements expanding or rolling window training:
    - Expanding: Train on all available past data
    - Rolling: Train on fixed window of past seasons

    Example:
        >>> validator = WalkForwardValidator(train_seasons=3, test_seasons=1)
        >>> results = validator.validate(
        ...     model=SpreadModel(),
        ...     df=training_data,
        ...     target_col="actual_spread",
        ...     feature_cols=feature_names,
        ...     season_col="season",
        ...     line_col="spread_line",
        ... )
        >>> print(results.summary())
    """

    def __init__(
        self,
        train_seasons: int = 3,
        test_seasons: int = 1,
        min_train_seasons: int = 2,
        expanding: bool = False,
        step_size: int = 1,
    ):
        """
        Initialize the validator.

        Args:
            train_seasons: Number of seasons for training window
            test_seasons: Number of seasons for testing (usually 1)
            min_train_seasons: Minimum training seasons required
            expanding: If True, use expanding window (all past data)
            step_size: Seasons to move forward between folds
        """
        self.train_seasons = train_seasons
        self.test_seasons = test_seasons
        self.min_train_seasons = min_train_seasons
        self.expanding = expanding
        self.step_size = step_size
        self.logger = logger.bind(component="walk_forward_validator")

    def get_fold_splits(
        self,
        seasons: list[int],
    ) -> list[tuple[list[int], list[int]]]:
        """
        Generate train/test season splits.

        Args:
            seasons: List of available seasons (sorted ascending)

        Returns:
            List of (train_seasons, test_seasons) tuples
        """
        seasons = sorted(seasons)
        splits = []

        # Determine starting point
        start_idx = self.min_train_seasons

        for test_start in range(start_idx, len(seasons), self.step_size):
            test_end = min(test_start + self.test_seasons, len(seasons))
            test = seasons[test_start:test_end]

            if self.expanding:
                # Use all prior seasons
                train = seasons[:test_start]
            else:
                # Use fixed window
                train_start = max(0, test_start - self.train_seasons)
                train = seasons[train_start:test_start]

            if len(train) >= self.min_train_seasons and len(test) > 0:
                splits.append((train, test))

        return splits

    def validate(
        self,
        model: BaseModel,
        df: pl.DataFrame,
        target_col: str,
        feature_cols: list[str],
        season_col: str = "season",
        line_col: Optional[str] = None,
        model_factory: Optional[Callable[[], BaseModel]] = None,
    ) -> ValidationResults:
        """
        Perform walk-forward validation.

        Args:
            model: Model instance (used as template if model_factory not provided)
            df: DataFrame with features, target, and season columns
            target_col: Name of target column
            feature_cols: List of feature column names
            season_col: Column containing season identifier
            line_col: Optional column with betting lines for ATS evaluation
            model_factory: Optional factory to create fresh model per fold

        Returns:
            ValidationResults with metrics from all folds
        """
        # Get available seasons
        seasons = sorted(df[season_col].unique().to_list())
        splits = self.get_fold_splits(seasons)

        if not splits:
            raise ValueError(
                f"Not enough seasons for validation. "
                f"Need at least {self.min_train_seasons + 1} seasons."
            )

        self.logger.info(
            f"Running walk-forward validation with {len(splits)} folds"
        )

        fold_results = []

        for fold_id, (train_seasons, test_seasons) in enumerate(splits):
            self.logger.debug(
                f"Fold {fold_id + 1}: Train {train_seasons} → Test {test_seasons}"
            )

            # Split data
            train_df = df.filter(pl.col(season_col).is_in(train_seasons))
            test_df = df.filter(pl.col(season_col).is_in(test_seasons))

            # Prepare features and targets
            X_train = train_df.select(feature_cols).to_numpy()
            y_train = train_df[target_col].to_numpy()
            X_test = test_df.select(feature_cols).to_numpy()
            y_test = test_df[target_col].to_numpy()

            # Get lines if available
            lines = None
            if line_col and line_col in test_df.columns:
                lines = test_df[line_col].to_numpy()

            # Create fresh model instance
            if model_factory:
                fold_model = model_factory()
            else:
                fold_model = model.__class__()

            # Train
            fold_model.train(X_train, y_train)

            # Predict
            predictions = fold_model.predict(X_test)

            # Evaluate
            metrics = fold_model.evaluate(X_test, y_test, lines)

            # Get feature importance
            try:
                feature_importance = fold_model.get_feature_importance()
            except (ValueError, AttributeError):
                feature_importance = {}

            fold_results.append(
                FoldResult(
                    fold_id=fold_id + 1,
                    train_seasons=train_seasons,
                    test_season=test_seasons[0],
                    metrics=metrics,
                    predictions=predictions,
                    actuals=y_test,
                    lines=lines,
                    feature_importance=feature_importance,
                )
            )

        results = ValidationResults(
            fold_results=fold_results,
            model_type=model.MODEL_TYPE,
        )

        self.logger.info(f"Validation complete: {results.summary()}")
        return results


class TimeSeriesSplit:
    """
    Time-series cross-validator for weekly NFL data.

    Similar to sklearn's TimeSeriesSplit but designed for NFL seasons
    where each game is a separate observation within a season.
    """

    def __init__(
        self,
        n_splits: int = 5,
        test_size: Optional[int] = None,
        gap: int = 0,
    ):
        """
        Initialize the splitter.

        Args:
            n_splits: Number of splits
            test_size: Fixed test set size (if None, grows with each fold)
            gap: Number of samples to exclude between train and test
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap

    def split(
        self,
        X: Union[np.ndarray, pl.DataFrame],
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ):
        """
        Generate train/test indices for each fold.

        Args:
            X: Feature matrix
            y: Target values (unused, for API compatibility)
            groups: Group labels (unused, for API compatibility)

        Yields:
            Tuple of (train_indices, test_indices)
        """
        n_samples = len(X)

        if self.test_size:
            test_size = self.test_size
        else:
            test_size = n_samples // (self.n_splits + 1)

        test_starts = range(
            test_size + self.gap,
            n_samples,
            (n_samples - test_size) // self.n_splits,
        )

        for test_start in list(test_starts)[: self.n_splits]:
            train_end = test_start - self.gap
            test_end = min(test_start + test_size, n_samples)

            train_indices = np.arange(train_end)
            test_indices = np.arange(test_start, test_end)

            yield train_indices, test_indices

    def get_n_splits(self) -> int:
        """Return the number of splits."""
        return self.n_splits


def evaluate_against_closing_line(
    predictions: np.ndarray,
    lines_at_prediction: np.ndarray,
    closing_lines: np.ndarray,
    actuals: np.ndarray,
) -> dict[str, float]:
    """
    Evaluate predictions against closing line value (CLV).

    CLV is a key indicator of long-term betting success.

    Args:
        predictions: Model predictions
        lines_at_prediction: Lines when predictions were made
        closing_lines: Final closing lines
        actuals: Actual game results

    Returns:
        Dictionary with CLV metrics
    """
    # CLV: How much better than closing line was the bet?
    # Positive CLV = got a better line than closing

    # For home team bets (prediction > line means bet home)
    home_bets = predictions > lines_at_prediction
    clv = np.where(
        home_bets,
        lines_at_prediction - closing_lines,  # Home: lower line is better
        closing_lines - lines_at_prediction,  # Away: higher line is better
    )

    # Calculate win rate for positive CLV bets
    positive_clv = clv > 0
    n_positive_clv = np.sum(positive_clv)

    if n_positive_clv > 0:
        # Calculate ATS results for positive CLV bets only
        home_covers = actuals > closing_lines
        correct = home_covers == home_bets
        positive_clv_win_rate = np.mean(correct[positive_clv])
    else:
        positive_clv_win_rate = 0.0

    return {
        "mean_clv": float(np.mean(clv)),
        "positive_clv_rate": float(np.mean(positive_clv)),
        "positive_clv_win_rate": positive_clv_win_rate,
        "n_positive_clv_bets": int(n_positive_clv),
    }


def calculate_calibration_curve(
    predicted_probs: np.ndarray,
    actual_outcomes: np.ndarray,
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate calibration curve for probability predictions.

    Args:
        predicted_probs: Predicted probabilities
        actual_outcomes: Binary actual outcomes (0 or 1)
        n_bins: Number of bins for calibration

    Returns:
        Tuple of (mean_predicted, mean_actual, counts) for each bin
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(predicted_probs, bin_edges[1:-1])

    mean_predicted = np.zeros(n_bins)
    mean_actual = np.zeros(n_bins)
    counts = np.zeros(n_bins)

    for i in range(n_bins):
        mask = bin_indices == i
        counts[i] = np.sum(mask)
        if counts[i] > 0:
            mean_predicted[i] = np.mean(predicted_probs[mask])
            mean_actual[i] = np.mean(actual_outcomes[mask])

    return mean_predicted, mean_actual, counts


def expected_calibration_error(
    predicted_probs: np.ndarray,
    actual_outcomes: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Calculate Expected Calibration Error (ECE).

    ECE measures how well predicted probabilities match actual frequencies.
    Lower is better, target < 0.05 for well-calibrated models.

    Args:
        predicted_probs: Predicted probabilities
        actual_outcomes: Binary actual outcomes
        n_bins: Number of bins

    Returns:
        ECE value (0 = perfectly calibrated)
    """
    mean_predicted, mean_actual, counts = calculate_calibration_curve(
        predicted_probs, actual_outcomes, n_bins
    )

    # ECE = weighted average of |predicted - actual| for each bin
    total = np.sum(counts)
    if total == 0:
        return 0.0

    weights = counts / total
    errors = np.abs(mean_predicted - mean_actual)

    # Only count non-empty bins
    mask = counts > 0
    return float(np.sum(weights[mask] * errors[mask]))


def brier_score(
    predicted_probs: np.ndarray,
    actual_outcomes: np.ndarray,
) -> float:
    """
    Calculate Brier Score for probability predictions.

    Brier Score = mean((predicted - actual)²)
    Lower is better, 0.25 is equivalent to random guessing for 50/50 events.

    Args:
        predicted_probs: Predicted probabilities
        actual_outcomes: Binary actual outcomes (0 or 1)

    Returns:
        Brier score
    """
    return float(np.mean((predicted_probs - actual_outcomes) ** 2))


def log_loss(
    predicted_probs: np.ndarray,
    actual_outcomes: np.ndarray,
    eps: float = 1e-15,
) -> float:
    """
    Calculate log loss (cross-entropy) for probability predictions.

    Args:
        predicted_probs: Predicted probabilities
        actual_outcomes: Binary actual outcomes
        eps: Small value to avoid log(0)

    Returns:
        Log loss value
    """
    probs = np.clip(predicted_probs, eps, 1 - eps)
    return float(
        -np.mean(
            actual_outcomes * np.log(probs)
            + (1 - actual_outcomes) * np.log(1 - probs)
        )
    )
