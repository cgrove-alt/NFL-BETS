"""
Base class for player prop prediction models.

Provides quantile regression for uncertainty estimation:
- Predict multiple quantiles (10th, 25th, 50th, 75th, 90th)
- Estimate full prediction distribution
- Calculate over/under probabilities for any line

Quantile regression is better than point prediction for props because:
- Props have asymmetric distributions (can't go below 0)
- Uncertainty varies by player and matchup
- We need probabilities, not just point estimates
"""
from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional, Union

import joblib
import numpy as np
import polars as pl
from loguru import logger
from scipy import stats
from sklearn.linear_model import QuantileRegressor
from sklearn.model_selection import train_test_split

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

from ..base import BaseModel, ModelMetadata, ModelMetrics, PredictionResult
from ..calibration import ProbabilityCalibrator


@dataclass
class PropPrediction:
    """Container for a single player prop prediction."""

    player_id: str
    player_name: str
    game_id: str
    prop_type: str
    team: str
    opponent: str

    # Distribution estimates
    predicted_value: float  # Median prediction
    prediction_mean: float
    prediction_std: float

    # Quantile predictions
    quantile_10: float
    quantile_25: float
    quantile_50: float  # Same as predicted_value
    quantile_75: float
    quantile_90: float

    # Betting context
    line: Optional[float] = None
    over_prob: Optional[float] = None
    under_prob: Optional[float] = None
    edge: Optional[float] = None

    # Metadata
    model_version: str = ""
    prediction_time: datetime = field(default_factory=datetime.now)

    @property
    def pick(self) -> Optional[str]:
        """Get the model's pick (over or under)."""
        if self.over_prob is None or self.under_prob is None:
            return None
        return "over" if self.over_prob > self.under_prob else "under"

    @property
    def confidence(self) -> Optional[float]:
        """Get confidence in the pick (0.5 to 1.0)."""
        if self.over_prob is None:
            return None
        return max(self.over_prob, 1 - self.over_prob)

    @property
    def iqr(self) -> float:
        """Interquartile range (uncertainty measure)."""
        return self.quantile_75 - self.quantile_25

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "player_id": self.player_id,
            "player_name": self.player_name,
            "game_id": self.game_id,
            "prop_type": self.prop_type,
            "team": self.team,
            "opponent": self.opponent,
            "predicted_value": self.predicted_value,
            "prediction_mean": self.prediction_mean,
            "prediction_std": self.prediction_std,
            "quantiles": {
                "10": self.quantile_10,
                "25": self.quantile_25,
                "50": self.quantile_50,
                "75": self.quantile_75,
                "90": self.quantile_90,
            },
            "line": self.line,
            "over_prob": self.over_prob,
            "under_prob": self.under_prob,
            "edge": self.edge,
            "pick": self.pick,
            "confidence": self.confidence,
            "model_version": self.model_version,
            "prediction_time": self.prediction_time.isoformat(),
        }


class BasePropModel(BaseModel):
    """
    Base class for player prop models using quantile regression.

    Subclasses should implement:
    - prop_type: The type of prop (passing_yards, rushing_yards, etc.)
    - position_filter: List of positions this model applies to
    - FEATURE_WEIGHTS: Optional feature weighting for this prop type

    Example:
        >>> class PassingYardsModel(BasePropModel):
        ...     prop_type = "passing_yards"
        ...     position_filter = ["QB"]
        ...
        >>> model = PassingYardsModel()
        >>> model.train(X_train, y_train)
        >>> prediction = model.predict_player(features, "Patrick Mahomes", ...)
    """

    MODEL_TYPE = "prop_quantile"
    VERSION = "1.0.0"

    # Override in subclasses
    prop_type: str = "generic"
    position_filter: list[str] = []

    # Quantiles to predict
    QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]

    # Feature weights (override in subclasses for prop-specific weighting)
    FEATURE_WEIGHTS: dict[str, float] = {}

    # Default model parameters
    MODEL_PARAMS = {
        "n_estimators": 300,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
    }

    def __init__(
        self,
        quantiles: Optional[list[float]] = None,
        model_params: Optional[dict] = None,
        use_calibration: bool = True,
    ):
        """
        Initialize the prop model.

        Args:
            quantiles: Quantiles to predict (default: [0.1, 0.25, 0.5, 0.75, 0.9])
            model_params: Model hyperparameters
            use_calibration: Whether to calibrate probabilities
        """
        super().__init__()

        self.quantiles = quantiles or self.QUANTILES
        self.model_params = {**self.MODEL_PARAMS, **(model_params or {})}
        self.use_calibration = use_calibration

        # Quantile models
        self._quantile_models: dict[float, Any] = {}

        # Calibrator
        self.calibrator: Optional[ProbabilityCalibrator] = None

        self.logger = logger.bind(model=f"prop_{self.prop_type}")

    def train(
        self,
        X: Union[pl.DataFrame, np.ndarray],
        y: Union[pl.Series, np.ndarray],
        validation_data: Optional[tuple] = None,
        fit_calibrator: bool = True,
        season_week_index: Optional[np.ndarray] = None,
    ) -> "BasePropModel":
        """
        Train quantile models for all target quantiles.

        Args:
            X: Feature matrix
            y: Target values (actual stats)
            validation_data: Optional (X_val, y_val) for calibration
            fit_calibrator: Whether to fit probability calibrator
            season_week_index: Optional temporal index for chronological sorting

        Returns:
            Self for method chaining
        """
        X_arr = self._prepare_features(X)
        y_arr = y.to_numpy() if isinstance(y, pl.Series) else np.asarray(y)

        # Sort chronologically if index provided
        if season_week_index is not None:
            sort_idx = np.argsort(season_week_index)
            X_arr = X_arr[sort_idx]
            y_arr = y_arr[sort_idx]
            self.logger.debug("Data sorted chronologically by season_week_index")

        # Apply feature weights if specified
        if self.FEATURE_WEIGHTS and len(self.feature_names) > 0:
            weights = np.array([
                self.FEATURE_WEIGHTS.get(f, 1.0) for f in self.feature_names
            ])
            X_arr = X_arr * weights

        self.logger.info(
            f"Training {self.prop_type} model on {len(y_arr)} samples"
        )

        # CRITICAL: Use chronological split (no shuffling) to prevent look-ahead bias
        # Split: 70% train, 15% validation, 15% calibration
        if validation_data is None:
            n_samples = len(y_arr)
            train_end = int(n_samples * 0.70)
            val_end = int(n_samples * 0.85)

            X_train = X_arr[:train_end]
            y_train = y_arr[:train_end]
            X_val = X_arr[train_end:val_end]
            y_val = y_arr[train_end:val_end]
            X_cal = X_arr[val_end:]
            y_cal = y_arr[val_end:]

            self.logger.info(
                f"Chronological split: train={len(y_train)}, val={len(y_val)}, cal={len(y_cal)}"
            )
        else:
            # External validation data provided - use it
            X_train, y_train = X_arr, y_arr
            X_val = self._prepare_features(validation_data[0])
            y_val = (
                validation_data[1].to_numpy()
                if isinstance(validation_data[1], pl.Series)
                else np.asarray(validation_data[1])
            )
            if self.FEATURE_WEIGHTS:
                weights = np.array([
                    self.FEATURE_WEIGHTS.get(f, 1.0) for f in self.feature_names
                ])
                X_val = X_val * weights
            # Use validation data for calibration too (backward compat)
            X_cal, y_cal = X_val, y_val

        # Train quantile models
        for q in self.quantiles:
            self._train_quantile_model(X_train, y_train, q)

        # Fit calibrator on CALIBRATION SET (not validation) to prevent leakage
        if fit_calibrator and self.use_calibration:
            self._fit_calibrator(X_cal, y_cal)

        self.is_fitted = True

        # Create metadata
        self.metadata = ModelMetadata(
            model_type=f"{self.MODEL_TYPE}_{self.prop_type}",
            model_version=self.VERSION,
            training_date=datetime.now(),
            training_seasons=[],
            n_training_samples=len(y_arr),
            feature_names=self.feature_names,
            hyperparameters={
                "quantiles": self.quantiles,
                "model_params": self.model_params,
                "feature_weights": self.FEATURE_WEIGHTS,
            },
            metrics=self.evaluate(X_val, y_val),
            calibrated=self.calibrator is not None,
        )

        self.logger.info(f"{self.prop_type} model training complete")
        return self

    def _train_quantile_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        quantile: float,
    ) -> None:
        """Train a model for a specific quantile."""
        self.logger.debug(f"Training quantile {quantile} model...")

        if HAS_LGB:
            # Use LightGBM with quantile objective
            model = lgb.LGBMRegressor(
                objective="quantile",
                alpha=quantile,
                **{k: v for k, v in self.model_params.items()
                   if k not in ["reg_alpha", "reg_lambda"]},
                verbose=-1,
            )
            model.fit(X, y)
        elif HAS_XGB:
            # Use XGBoost with quantile loss
            model = xgb.XGBRegressor(
                objective="reg:quantileerror",
                quantile_alpha=quantile,
                **self.model_params,
            )
            model.fit(X, y)
        else:
            # Fallback to sklearn's QuantileRegressor (slower)
            model = QuantileRegressor(
                quantile=quantile,
                alpha=0.1,
                solver="highs",
            )
            model.fit(X, y)

        self._quantile_models[quantile] = model

    def _fit_calibrator(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> None:
        """Fit probability calibrator using validation data."""
        self.logger.debug("Fitting probability calibrator...")

        # Generate predictions at various synthetic lines
        median_preds = self._quantile_models[0.5].predict(X_val)

        all_probs = []
        all_outcomes = []

        # Test multiple lines around predictions
        for pct in [0.7, 0.85, 1.0, 1.15, 1.3]:
            synthetic_lines = median_preds * pct
            raw_probs = self._calculate_over_probability(X_val, synthetic_lines)
            outcomes = (y_val > synthetic_lines).astype(float)
            all_probs.extend(raw_probs)
            all_outcomes.extend(outcomes)

        self.calibrator = ProbabilityCalibrator(method="isotonic")
        self.calibrator.fit(np.array(all_probs), np.array(all_outcomes))

    def predict(
        self,
        X: Union[pl.DataFrame, np.ndarray],
    ) -> np.ndarray:
        """
        Predict median values.

        Args:
            X: Feature matrix

        Returns:
            Array of median predictions
        """
        self._validate_fitted()
        X_arr = self._prepare_features(X)

        if self.FEATURE_WEIGHTS and len(self.feature_names) > 0:
            weights = np.array([
                self.FEATURE_WEIGHTS.get(f, 1.0) for f in self.feature_names
            ])
            X_arr = X_arr * weights

        return self._quantile_models[0.5].predict(X_arr)

    def predict_proba(
        self,
        X: Union[pl.DataFrame, np.ndarray],
        lines: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Predict over probabilities.

        Args:
            X: Feature matrix
            lines: Lines to evaluate (if None, uses median predictions)

        Returns:
            Array of over probabilities
        """
        self._validate_fitted()
        X_arr = self._prepare_features(X)

        if self.FEATURE_WEIGHTS and len(self.feature_names) > 0:
            weights = np.array([
                self.FEATURE_WEIGHTS.get(f, 1.0) for f in self.feature_names
            ])
            X_arr = X_arr * weights

        if lines is None:
            lines = self._quantile_models[0.5].predict(X_arr)

        raw_probs = self._calculate_over_probability(X_arr, lines)

        if self.calibrator is not None:
            return self.calibrator.calibrate(raw_probs)

        return raw_probs

    def predict_quantiles(
        self,
        X: Union[pl.DataFrame, np.ndarray],
        enforce_monotonicity: bool = True,
    ) -> dict[float, np.ndarray]:
        """
        Predict all quantiles with optional monotonicity enforcement.

        Args:
            X: Feature matrix
            enforce_monotonicity: If True, fix quantile crossing issues

        Returns:
            Dictionary mapping quantile to predictions
        """
        self._validate_fitted()
        X_arr = self._prepare_features(X)

        if self.FEATURE_WEIGHTS and len(self.feature_names) > 0:
            weights = np.array([
                self.FEATURE_WEIGHTS.get(f, 1.0) for f in self.feature_names
            ])
            X_arr = X_arr * weights

        # Get raw predictions
        raw_preds = {q: model.predict(X_arr) for q, model in self._quantile_models.items()}

        if not enforce_monotonicity:
            return raw_preds

        # Fix quantile crossing by enforcing monotonicity
        return self._enforce_quantile_monotonicity(raw_preds)

    def _enforce_quantile_monotonicity(
        self,
        quantile_preds: dict[float, np.ndarray],
    ) -> dict[float, np.ndarray]:
        """
        Enforce monotonicity in quantile predictions.

        Quantile crossing occurs when q_lower > q_upper for some samples.
        This fixes violations by sorting predictions for each sample.

        Args:
            quantile_preds: Raw quantile predictions

        Returns:
            Monotonicity-enforced predictions
        """
        n_samples = len(quantile_preds[self.quantiles[0]])
        sorted_qs = sorted(self.quantiles)

        # Stack all predictions: shape (n_quantiles, n_samples)
        stacked = np.vstack([quantile_preds[q] for q in sorted_qs])

        # For each sample, sort the quantile values to enforce monotonicity
        for i in range(n_samples):
            sample_vals = stacked[:, i]

            # Check if monotonicity is violated
            if not np.all(np.diff(sample_vals) >= 0):
                # Sort values to enforce monotonicity
                stacked[:, i] = np.sort(sample_vals)

        # Unpack back to dictionary
        fixed_preds = {}
        for j, q in enumerate(sorted_qs):
            fixed_preds[q] = stacked[j, :]

        return fixed_preds

    def predict_distribution(
        self,
        X: Union[pl.DataFrame, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Estimate mean and std from quantile predictions.

        Args:
            X: Feature matrix

        Returns:
            Tuple of (mean, std) arrays
        """
        quantile_preds = self.predict_quantiles(X)

        # Use median as mean estimate
        mean = quantile_preds[0.5]

        # Estimate std from IQR: std ≈ IQR / 1.35
        iqr = quantile_preds[0.75] - quantile_preds[0.25]
        std = iqr / 1.35

        return mean, std

    def _calculate_over_probability(
        self,
        X: np.ndarray,
        lines: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate raw over probability from quantile predictions.

        Uses linear interpolation between quantiles.
        """
        quantile_preds = {q: self._quantile_models[q].predict(X) for q in self.quantiles}

        probs = np.zeros(len(X))

        for i in range(len(X)):
            line = lines[i]
            preds = {q: quantile_preds[q][i] for q in self.quantiles}

            # Find where line falls in quantile predictions
            sorted_q = sorted(self.quantiles)

            if line <= preds[sorted_q[0]]:
                # Line below 10th percentile
                probs[i] = 0.95
            elif line >= preds[sorted_q[-1]]:
                # Line above 90th percentile
                probs[i] = 0.05
            else:
                # Interpolate between quantiles
                for j in range(len(sorted_q) - 1):
                    q_low, q_high = sorted_q[j], sorted_q[j + 1]
                    p_low, p_high = preds[q_low], preds[q_high]

                    if p_low <= line <= p_high:
                        # Linear interpolation
                        frac = (line - p_low) / (p_high - p_low) if p_high > p_low else 0.5
                        under_prob = q_low + frac * (q_high - q_low)
                        probs[i] = 1 - under_prob
                        break

        return probs

    def predict_player(
        self,
        features: Union[dict, pl.DataFrame, np.ndarray],
        player_id: str,
        player_name: str,
        game_id: str,
        team: str,
        opponent: str,
        line: Optional[float] = None,
    ) -> PropPrediction:
        """
        Make a single player prop prediction.

        Args:
            features: Feature values for the player/game
            player_id: Player's unique identifier
            player_name: Player's name
            game_id: Game identifier
            team: Player's team
            opponent: Opposing team
            line: Optional betting line

        Returns:
            PropPrediction with all details
        """
        self._validate_fitted()

        # Prepare features
        if isinstance(features, dict):
            X = np.array([[features.get(f, 0.0) for f in self.feature_names]])
        elif isinstance(features, pl.DataFrame):
            X = features.select(self.feature_names).to_numpy()
        else:
            X = np.atleast_2d(features)

        if self.FEATURE_WEIGHTS and len(self.feature_names) > 0:
            weights = np.array([
                self.FEATURE_WEIGHTS.get(f, 1.0) for f in self.feature_names
            ])
            X = X * weights

        # Get quantile predictions
        quantile_preds = {q: float(self._quantile_models[q].predict(X)[0])
                         for q in self.quantiles}

        # Estimate mean and std
        mean = quantile_preds[0.5]
        iqr = quantile_preds[0.75] - quantile_preds[0.25]
        std = iqr / 1.35

        # Calculate over/under probabilities if line provided
        over_prob = None
        under_prob = None
        edge = None

        if line is not None:
            over_prob = float(self.predict_proba(X, np.array([line]))[0])
            under_prob = 1 - over_prob
            # Edge relative to 50/50
            edge = abs(over_prob - 0.5)

        return PropPrediction(
            player_id=player_id,
            player_name=player_name,
            game_id=game_id,
            prop_type=self.prop_type,
            team=team,
            opponent=opponent,
            predicted_value=quantile_preds[0.5],
            prediction_mean=mean,
            prediction_std=std,
            quantile_10=quantile_preds.get(0.1, quantile_preds[0.25] - iqr),
            quantile_25=quantile_preds[0.25],
            quantile_50=quantile_preds[0.5],
            quantile_75=quantile_preds[0.75],
            quantile_90=quantile_preds.get(0.9, quantile_preds[0.75] + iqr),
            line=line,
            over_prob=over_prob,
            under_prob=under_prob,
            edge=edge,
            model_version=self.VERSION,
        )

    def predict_player_with_uncertainty(
        self,
        features: Union[dict, pl.DataFrame, np.ndarray],
        player_id: str,
        player_name: str,
        game_id: str,
        team: str,
        opponent: str,
        line: Optional[float] = None,
        uncertainty_multiplier: float = 1.0,
    ) -> PropPrediction:
        """
        Make a player prop prediction with adjusted uncertainty.

        Widens prediction intervals based on uncertainty multiplier.
        Used for:
        - Questionable/Doubtful players (higher uncertainty)
        - Backup players stepping into starter roles
        - Players with injured teammates (usage uncertainty)

        Args:
            features: Feature values for the player/game
            player_id: Player's unique identifier
            player_name: Player's name
            game_id: Game identifier
            team: Player's team
            opponent: Opposing team
            line: Optional betting line
            uncertainty_multiplier: Multiplier for prediction intervals
                - 1.0 = normal (healthy starter)
                - 1.5 = moderate (Questionable or backup)
                - 2.0+ = high (Doubtful or limited backup data)

        Returns:
            PropPrediction with widened intervals
        """
        self._validate_fitted()

        # Get base prediction
        base_pred = self.predict_player(
            features=features,
            player_id=player_id,
            player_name=player_name,
            game_id=game_id,
            team=team,
            opponent=opponent,
            line=line,
        )

        # If no uncertainty adjustment needed, return base
        if uncertainty_multiplier == 1.0:
            return base_pred

        # Widen the distribution from the median
        median = base_pred.quantile_50

        # Adjust quantiles by multiplying the distance from median
        adjusted_q10 = median - (median - base_pred.quantile_10) * uncertainty_multiplier
        adjusted_q25 = median - (median - base_pred.quantile_25) * uncertainty_multiplier
        adjusted_q75 = median + (base_pred.quantile_75 - median) * uncertainty_multiplier
        adjusted_q90 = median + (base_pred.quantile_90 - median) * uncertainty_multiplier

        # Ensure non-negative values (can't have negative yards/receptions)
        adjusted_q10 = max(0.0, adjusted_q10)
        adjusted_q25 = max(0.0, adjusted_q25)

        # Adjust std
        adjusted_std = base_pred.prediction_std * uncertainty_multiplier

        # Recalculate over/under probabilities with wider distribution
        # The wider distribution means probabilities move toward 0.5
        over_prob = base_pred.over_prob
        under_prob = base_pred.under_prob
        edge = base_pred.edge

        if line is not None and over_prob is not None:
            # Use normal distribution approximation with wider std
            # This makes the probability closer to 0.5 as uncertainty increases
            z_score = (line - median) / adjusted_std if adjusted_std > 0 else 0
            over_prob = 1 - stats.norm.cdf(z_score)
            under_prob = 1 - over_prob
            edge = abs(over_prob - 0.5)

            # Apply calibration if available
            if self.calibrator is not None:
                over_prob = float(self.calibrator.calibrate(np.array([over_prob]))[0])
                under_prob = 1 - over_prob
                edge = abs(over_prob - 0.5)

        return PropPrediction(
            player_id=player_id,
            player_name=player_name,
            game_id=game_id,
            prop_type=self.prop_type,
            team=team,
            opponent=opponent,
            predicted_value=median,
            prediction_mean=median,
            prediction_std=adjusted_std,
            quantile_10=adjusted_q10,
            quantile_25=adjusted_q25,
            quantile_50=median,
            quantile_75=adjusted_q75,
            quantile_90=adjusted_q90,
            line=line,
            over_prob=over_prob,
            under_prob=under_prob,
            edge=edge,
            model_version=self.VERSION,
        )

    def get_optimal_line(
        self,
        X: Union[pl.DataFrame, np.ndarray],
        target_prob: float = 0.5,
    ) -> np.ndarray:
        """
        Find the line where over probability equals target.

        Args:
            X: Feature matrix
            target_prob: Target over probability

        Returns:
            Array of optimal lines
        """
        quantile_preds = self.predict_quantiles(X)

        # Target quantile is 1 - target_prob (e.g., 0.5 over prob = 0.5 quantile)
        target_quantile = 1 - target_prob

        # Interpolate to find the value at target quantile
        sorted_q = sorted(self.quantiles)
        n_samples = len(X) if hasattr(X, "__len__") else 1

        results = np.zeros(n_samples)

        for i in range(n_samples):
            for j in range(len(sorted_q) - 1):
                q_low, q_high = sorted_q[j], sorted_q[j + 1]

                if q_low <= target_quantile <= q_high:
                    p_low = quantile_preds[q_low][i]
                    p_high = quantile_preds[q_high][i]
                    frac = (target_quantile - q_low) / (q_high - q_low)
                    results[i] = p_low + frac * (p_high - p_low)
                    break
            else:
                # Default to median
                results[i] = quantile_preds[0.5][i]

        return results

    def save(self, path: Union[str, Path]) -> None:
        """Save the model to disk."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        save_dict = {
            "model_type": self.MODEL_TYPE,
            "prop_type": self.prop_type,
            "version": self.VERSION,
            "quantile_models": self._quantile_models,
            "quantiles": self.quantiles,
            "calibrator": self.calibrator,
            "feature_names": self.feature_names,
            "feature_weights": self.FEATURE_WEIGHTS,
            "model_params": self.model_params,
            "metadata": self.metadata,
            "is_fitted": self.is_fitted,
        }

        joblib.dump(save_dict, path)
        self.logger.info(f"Model saved to {path}")

    def load(self, path: Union[str, Path]) -> "BasePropModel":
        """Load a model from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        save_dict = joblib.load(path)

        if save_dict.get("prop_type") != self.prop_type:
            raise ValueError(
                f"Prop type mismatch: expected {self.prop_type}, "
                f"got {save_dict.get('prop_type')}"
            )

        self._quantile_models = save_dict["quantile_models"]
        self.quantiles = save_dict["quantiles"]
        self.calibrator = save_dict.get("calibrator")
        self.feature_names = save_dict["feature_names"]
        self.FEATURE_WEIGHTS = save_dict.get("feature_weights", {})
        self.model_params = save_dict.get("model_params", self.MODEL_PARAMS)
        self.metadata = save_dict.get("metadata")
        self.is_fitted = save_dict.get("is_fitted", True)

        self.logger.info(f"Model loaded from {path}")
        return self
