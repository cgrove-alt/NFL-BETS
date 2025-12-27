"""
Abstract base classes for ML models.

Provides common interface for all prediction models:
- Training and prediction methods
- Model persistence (save/load)
- Feature importance extraction
- Prediction confidence intervals
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
import logging
from pathlib import Path
from typing import Any, Optional, Union

import joblib
import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    """Container for model evaluation metrics."""

    # Regression metrics
    mae: float = 0.0
    rmse: float = 0.0
    r2: float = 0.0

    # Classification metrics
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0

    # Calibration metrics (basic)
    calibration_error: float = 0.0
    brier_score: float = 0.0

    # Calibration metrics (detailed)
    log_loss: Optional[float] = None  # Cross-entropy loss
    ece: Optional[float] = None  # Expected Calibration Error
    mce: Optional[float] = None  # Maximum Calibration Error

    # Betting-specific metrics
    ats_wins: int = 0
    ats_losses: int = 0
    ats_pushes: int = 0

    # Sample info
    n_samples: int = 0
    evaluation_date: datetime = field(default_factory=datetime.now)

    @property
    def ats_record(self) -> tuple[int, int, int]:
        """Get ATS record as (wins, losses, pushes)."""
        return (self.ats_wins, self.ats_losses, self.ats_pushes)

    @property
    def ats_win_rate(self) -> float:
        """Calculate ATS win rate (excluding pushes)."""
        total = self.ats_wins + self.ats_losses
        if total == 0:
            return 0.0
        return self.ats_wins / total

    @property
    def ats_roi(self) -> float:
        """
        Calculate ROI assuming -110 juice on all bets.

        Win = +100, Loss = -110
        ROI = (wins * 100 - losses * 110) / (total * 110)
        """
        total = self.ats_wins + self.ats_losses
        if total == 0:
            return 0.0
        profit = self.ats_wins * 100 - self.ats_losses * 110
        return profit / (total * 110)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "mae": self.mae,
            "rmse": self.rmse,
            "r2": self.r2,
            "accuracy": self.accuracy,
            "calibration_error": self.calibration_error,
            "brier_score": self.brier_score,
            "log_loss": self.log_loss,
            "ece": self.ece,
            "mce": self.mce,
            "ats_record": self.ats_record,
            "ats_win_rate": self.ats_win_rate,
            "ats_roi": self.ats_roi,
            "n_samples": self.n_samples,
            "evaluation_date": self.evaluation_date.isoformat(),
        }


@dataclass
class PredictionResult:
    """Container for model predictions."""

    # Point prediction
    prediction: np.ndarray

    # Uncertainty estimates
    confidence_lower: Optional[np.ndarray] = None
    confidence_upper: Optional[np.ndarray] = None
    std: Optional[np.ndarray] = None

    # Probability estimates (for classification or cover probability)
    probability: Optional[np.ndarray] = None

    # Metadata
    model_version: str = ""
    prediction_date: datetime = field(default_factory=datetime.now)
    feature_names: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "prediction": self.prediction.tolist(),
            "confidence_lower": (
                self.confidence_lower.tolist() if self.confidence_lower is not None else None
            ),
            "confidence_upper": (
                self.confidence_upper.tolist() if self.confidence_upper is not None else None
            ),
            "probability": (
                self.probability.tolist() if self.probability is not None else None
            ),
            "model_version": self.model_version,
            "prediction_date": self.prediction_date.isoformat(),
        }


@dataclass
class ModelMetadata:
    """Metadata about a trained model."""

    model_type: str
    model_version: str
    training_date: datetime
    training_seasons: list[int]
    n_training_samples: int
    feature_names: list[str]
    hyperparameters: dict[str, Any]
    metrics: ModelMetrics
    calibrated: bool = False
    data_cutoff_date: Optional[datetime] = None  # Date of most recent game in training data
    calibration_diagnostics: Optional[dict[str, Any]] = None  # ECE, MCE, Brier, log_loss, etc.

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model_type": self.model_type,
            "model_version": self.model_version,
            "training_date": self.training_date.isoformat(),
            "training_seasons": self.training_seasons,
            "n_training_samples": self.n_training_samples,
            "feature_names": self.feature_names,
            "hyperparameters": self.hyperparameters,
            "metrics": self.metrics.to_dict(),
            "calibrated": self.calibrated,
            "data_cutoff_date": self.data_cutoff_date.isoformat() if self.data_cutoff_date else None,
            "calibration_diagnostics": self.calibration_diagnostics,
        }

    def is_stale(self, latest_game_date: datetime) -> bool:
        """Check if model is stale (trained before latest available data)."""
        if self.data_cutoff_date is None:
            return True  # Unknown cutoff, assume stale
        return self.data_cutoff_date < latest_game_date


class BaseModel(ABC):
    """
    Abstract base class for all prediction models.

    Provides common interface for training, prediction, persistence,
    and feature importance analysis.

    Example:
        >>> model = SpreadModel()
        >>> model.train(X_train, y_train)
        >>> predictions = model.predict(X_test)
        >>> model.save(Path("models/spread_v1.joblib"))
    """

    MODEL_TYPE: str = "base"
    VERSION: str = "1.0.0"

    def __init__(self):
        self.is_fitted: bool = False
        self.feature_names: list[str] = []
        self.metadata: Optional[ModelMetadata] = None
        self._model = None

    @abstractmethod
    def train(
        self,
        X: Union[pl.DataFrame, np.ndarray],
        y: Union[pl.Series, np.ndarray],
        validation_data: Optional[tuple] = None,
        **kwargs,
    ) -> "BaseModel":
        """
        Train the model on the provided data.

        Args:
            X: Feature matrix
            y: Target values
            validation_data: Optional (X_val, y_val) for early stopping
            **kwargs: Additional training parameters

        Returns:
            Self for method chaining
        """
        pass

    @abstractmethod
    def predict(
        self,
        X: Union[pl.DataFrame, np.ndarray],
    ) -> np.ndarray:
        """
        Make predictions on new data.

        Args:
            X: Feature matrix

        Returns:
            Array of predictions
        """
        pass

    @abstractmethod
    def predict_proba(
        self,
        X: Union[pl.DataFrame, np.ndarray],
    ) -> np.ndarray:
        """
        Predict probabilities (for classification or cover probability).

        Args:
            X: Feature matrix

        Returns:
            Array of probabilities
        """
        pass

    def predict_with_confidence(
        self,
        X: Union[pl.DataFrame, np.ndarray],
        confidence_level: float = 0.90,
    ) -> PredictionResult:
        """
        Make predictions with confidence intervals.

        Args:
            X: Feature matrix
            confidence_level: Confidence level for intervals (default 0.90)

        Returns:
            PredictionResult with predictions and uncertainty estimates
        """
        predictions = self.predict(X)
        return PredictionResult(
            prediction=predictions,
            model_version=self.VERSION,
            feature_names=self.feature_names,
        )

    def evaluate(
        self,
        X: Union[pl.DataFrame, np.ndarray],
        y: Union[pl.Series, np.ndarray],
        lines: Optional[np.ndarray] = None,
    ) -> ModelMetrics:
        """
        Evaluate model performance on test data.

        Args:
            X: Feature matrix
            y: True target values
            lines: Optional betting lines for ATS evaluation

        Returns:
            ModelMetrics with evaluation results
        """
        y_arr = y.to_numpy() if isinstance(y, pl.Series) else y
        predictions = self.predict(X)

        # Calculate regression metrics
        mae = np.mean(np.abs(predictions - y_arr))
        rmse = np.sqrt(np.mean((predictions - y_arr) ** 2))
        ss_res = np.sum((y_arr - predictions) ** 2)
        ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        metrics = ModelMetrics(
            mae=mae,
            rmse=rmse,
            r2=r2,
            n_samples=len(y_arr),
        )

        # Calculate ATS metrics if lines provided
        if lines is not None:
            metrics = self._calculate_ats_metrics(predictions, y_arr, lines, metrics)

        return metrics

    def _calculate_ats_metrics(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        lines: np.ndarray,
        metrics: ModelMetrics,
    ) -> ModelMetrics:
        """Calculate against-the-spread betting metrics."""
        # Prediction: positive means home team should cover
        # Line: negative means home team is favored (e.g., -7 means home wins by 7)
        # Actual: home_score - away_score

        # Home covers if actual margin > line (e.g., if line is -7, home needs to win by 8+)
        home_covers = actuals > lines
        model_picks_home = predictions > lines

        # Calculate wins, losses, pushes
        correct = home_covers == model_picks_home
        pushes = np.isclose(actuals, lines, atol=0.5)

        metrics.ats_wins = int(np.sum(correct & ~pushes))
        metrics.ats_losses = int(np.sum(~correct & ~pushes))
        metrics.ats_pushes = int(np.sum(pushes))

        return metrics

    def get_feature_importance(self) -> dict[str, float]:
        """
        Get feature importance scores.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")

        if not hasattr(self._model, "feature_importances_"):
            logger.warning(f"{self.MODEL_TYPE} model does not support feature importance")
            return {}

        importances = self._model.feature_importances_
        return dict(zip(self.feature_names, importances))

    def get_top_features(self, n: int = 10) -> list[tuple[str, float]]:
        """
        Get top N most important features.

        Args:
            n: Number of features to return

        Returns:
            List of (feature_name, importance) tuples sorted by importance
        """
        importance = self.get_feature_importance()
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        return sorted_features[:n]

    def save(self, path: Union[str, Path]) -> None:
        """
        Save the trained model to disk.

        Args:
            path: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        save_dict = {
            "model": self._model,
            "metadata": self.metadata,
            "feature_names": self.feature_names,
            "is_fitted": self.is_fitted,
            "model_type": self.MODEL_TYPE,
            "version": self.VERSION,
        }

        joblib.dump(save_dict, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: Union[str, Path]) -> "BaseModel":
        """
        Load a trained model from disk.

        Args:
            path: Path to the saved model

        Returns:
            Self for method chaining
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        save_dict = joblib.load(path)

        # Validate model type
        if save_dict.get("model_type") != self.MODEL_TYPE:
            raise ValueError(
                f"Model type mismatch: expected {self.MODEL_TYPE}, "
                f"got {save_dict.get('model_type')}"
            )

        self._model = save_dict["model"]
        self.metadata = save_dict.get("metadata")
        self.feature_names = save_dict.get("feature_names", [])
        self.is_fitted = save_dict.get("is_fitted", True)

        logger.info(f"Model loaded from {path}")
        return self

    def _prepare_features(
        self,
        X: Union[pl.DataFrame, np.ndarray],
    ) -> np.ndarray:
        """
        Prepare features for prediction.

        Args:
            X: Feature matrix

        Returns:
            NumPy array of features
        """
        if isinstance(X, pl.DataFrame):
            # Store feature names if not already set
            if not self.feature_names:
                self.feature_names = X.columns
            return X.to_numpy()
        return X

    def _validate_fitted(self) -> None:
        """Raise error if model is not fitted."""
        if not self.is_fitted:
            raise ValueError(
                f"{self.MODEL_TYPE} model must be trained before making predictions"
            )


class EnsembleModel(BaseModel):
    """
    Base class for ensemble models combining multiple base models.

    Supports weighted averaging of predictions from multiple models.
    """

    MODEL_TYPE: str = "ensemble"

    def __init__(
        self,
        models: Optional[list[BaseModel]] = None,
        weights: Optional[list[float]] = None,
    ):
        super().__init__()
        self.models: list[BaseModel] = models or []
        self.weights: list[float] = weights or []

    def add_model(self, model: BaseModel, weight: float = 1.0) -> None:
        """Add a model to the ensemble."""
        self.models.append(model)
        self.weights.append(weight)

    def _normalize_weights(self) -> list[float]:
        """Normalize weights to sum to 1."""
        total = sum(self.weights)
        if total == 0:
            return [1.0 / len(self.weights)] * len(self.weights)
        return [w / total for w in self.weights]

    def predict(self, X: Union[pl.DataFrame, np.ndarray]) -> np.ndarray:
        """Make ensemble predictions using weighted average."""
        if not self.models:
            raise ValueError("No models in ensemble")

        weights = self._normalize_weights()
        predictions = np.zeros(len(X) if hasattr(X, "__len__") else 1)

        for model, weight in zip(self.models, weights):
            predictions += weight * model.predict(X)

        return predictions

    def predict_proba(self, X: Union[pl.DataFrame, np.ndarray]) -> np.ndarray:
        """Make ensemble probability predictions using weighted average."""
        if not self.models:
            raise ValueError("No models in ensemble")

        weights = self._normalize_weights()
        probabilities = np.zeros(len(X) if hasattr(X, "__len__") else 1)

        for model, weight in zip(self.models, weights):
            probabilities += weight * model.predict_proba(X)

        return probabilities


class QuantileModel(BaseModel):
    """
    Base class for quantile regression models.

    Predicts multiple quantiles to estimate prediction distribution.
    """

    MODEL_TYPE: str = "quantile"
    QUANTILES: list[float] = [0.1, 0.25, 0.5, 0.75, 0.9]

    def __init__(self, quantiles: Optional[list[float]] = None):
        super().__init__()
        self.quantiles = quantiles or self.QUANTILES
        self._quantile_models: dict[float, Any] = {}

    def predict_quantiles(
        self,
        X: Union[pl.DataFrame, np.ndarray],
    ) -> dict[float, np.ndarray]:
        """
        Predict multiple quantiles.

        Args:
            X: Feature matrix

        Returns:
            Dictionary mapping quantile to predictions
        """
        self._validate_fitted()
        X_arr = self._prepare_features(X)

        results = {}
        for q in self.quantiles:
            if q in self._quantile_models:
                results[q] = self._quantile_models[q].predict(X_arr)
        return results

    def predict_distribution(
        self,
        X: Union[pl.DataFrame, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Estimate mean and standard deviation from quantile predictions.

        Args:
            X: Feature matrix

        Returns:
            Tuple of (mean, std) arrays
        """
        quantile_preds = self.predict_quantiles(X)

        if 0.5 in quantile_preds:
            mean = quantile_preds[0.5]
        else:
            mean = np.mean(list(quantile_preds.values()), axis=0)

        # Estimate std from IQR: std ≈ IQR / 1.35
        if 0.25 in quantile_preds and 0.75 in quantile_preds:
            iqr = quantile_preds[0.75] - quantile_preds[0.25]
            std = iqr / 1.35
        else:
            # Fallback: use range of quantiles
            values = list(quantile_preds.values())
            std = (np.max(values, axis=0) - np.min(values, axis=0)) / 4

        return mean, std

    def predict_over_probability(
        self,
        X: Union[pl.DataFrame, np.ndarray],
        line: float,
    ) -> np.ndarray:
        """
        Estimate probability of going over a given line.

        Args:
            X: Feature matrix
            line: The line to compare against

        Returns:
            Array of probabilities
        """
        mean, std = self.predict_distribution(X)

        # Assume normal distribution for probability estimation
        # P(Y > line) = 1 - Phi((line - mean) / std)
        from scipy import stats

        z_scores = (line - mean) / np.maximum(std, 0.001)
        return 1 - stats.norm.cdf(z_scores)
