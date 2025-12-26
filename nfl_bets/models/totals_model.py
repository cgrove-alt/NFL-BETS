"""
Quantile regression model for NFL game totals (Over/Under) predictions.

Predicts the total combined score of both teams:
- Uses quantile regression for full distribution
- Calculates over/under probabilities for any line
- Provides uncertainty estimates

Quantile regression is better than point prediction for totals because:
- Score distributions can vary significantly
- Uncertainty matters for betting decisions
- We need probabilities, not just point estimates
"""
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

import joblib
import numpy as np
import polars as pl
from loguru import logger
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    xgb = None

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    lgb = None

from .base import BaseModel, ModelMetadata, ModelMetrics, PredictionResult
from .calibration import ProbabilityCalibrator


@dataclass
class TotalsPrediction:
    """Container for a single totals prediction."""

    game_id: str
    home_team: str
    away_team: str

    # Point prediction
    predicted_total: float  # Median prediction
    prediction_std: float

    # Quantile predictions
    quantile_10: float
    quantile_25: float
    quantile_50: float  # Same as predicted_total
    quantile_75: float
    quantile_90: float

    # Over/Under probabilities (for Vegas line)
    line: Optional[float] = None
    over_prob: Optional[float] = None
    under_prob: Optional[float] = None

    # Betting context
    over_odds: Optional[int] = None  # American odds
    under_odds: Optional[int] = None
    over_implied_prob: Optional[float] = None
    under_implied_prob: Optional[float] = None
    over_edge: Optional[float] = None
    under_edge: Optional[float] = None

    # Recommendation
    pick: Optional[str] = None  # "OVER" or "UNDER"
    edge: Optional[float] = None  # Edge on the pick
    bet_confidence: Optional[str] = None  # "LOW", "MEDIUM", "HIGH"

    # Metadata
    model_version: str = ""
    prediction_time: datetime = None

    def __post_init__(self):
        if self.prediction_time is None:
            self.prediction_time = datetime.now()
        # Set pick based on highest edge
        if self.over_edge is not None and self.under_edge is not None:
            if self.over_edge > self.under_edge and self.over_edge > 0:
                self.pick = "OVER"
                self.edge = self.over_edge
            elif self.under_edge > self.over_edge and self.under_edge > 0:
                self.pick = "UNDER"
                self.edge = self.under_edge

            # Set confidence level
            if self.edge:
                if self.edge >= 0.08:
                    self.bet_confidence = "HIGH"
                elif self.edge >= 0.04:
                    self.bet_confidence = "MEDIUM"
                else:
                    self.bet_confidence = "LOW"

    @property
    def iqr(self) -> float:
        """Interquartile range (uncertainty measure)."""
        return self.quantile_75 - self.quantile_25

    @property
    def confidence(self) -> Optional[float]:
        """Confidence in pick (0.5 to 1.0)."""
        if self.over_prob is None:
            return None
        return max(self.over_prob, 1 - self.over_prob)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "game_id": self.game_id,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "predicted_total": self.predicted_total,
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
            "over_odds": self.over_odds,
            "under_odds": self.under_odds,
            "over_implied_prob": self.over_implied_prob,
            "under_implied_prob": self.under_implied_prob,
            "over_edge": self.over_edge,
            "under_edge": self.under_edge,
            "pick": self.pick,
            "edge": self.edge,
            "bet_confidence": self.bet_confidence,
            "confidence": self.confidence,
            "iqr": self.iqr,
            "model_version": self.model_version,
            "prediction_time": self.prediction_time.isoformat(),
        }


class TotalsModel(BaseModel):
    """
    Quantile regression model for NFL game totals prediction.

    Uses LightGBM with quantile objectives to predict the full distribution
    of possible game totals, enabling accurate over/under probability
    calculation for any line.

    Example:
        >>> model = TotalsModel()
        >>> model.train(X_train, y_train)
        >>> prediction = model.predict_game(
        ...     features, game_id="2024_01_KC_BAL",
        ...     home_team="BAL", away_team="KC",
        ...     line=47.5, over_odds=-110, under_odds=-110
        ... )
        >>> print(f"Predicted total: {prediction.predicted_total:.1f}")
        >>> print(f"Over prob: {prediction.over_prob:.1%}")
    """

    MODEL_TYPE = "totals_quantile"
    VERSION = "1.0.0"

    # Quantiles to predict
    QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]

    # Default model parameters
    MODEL_PARAMS = {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 20,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    def __init__(
        self,
        quantiles: Optional[list[float]] = None,
        model_params: Optional[dict] = None,
        use_calibration: bool = True,
    ):
        """
        Initialize the totals model.

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

        # Track prediction statistics
        self._residual_std: float = 10.0  # Default NFL totals std

        self.logger = logger.bind(model="totals_quantile")

    def train(
        self,
        X: Union[pl.DataFrame, np.ndarray],
        y: Union[pl.Series, np.ndarray],
        validation_data: Optional[tuple] = None,
        fit_calibrator: bool = True,
    ) -> "TotalsModel":
        """
        Train quantile models for all target quantiles.

        Args:
            X: Feature matrix
            y: Target values (total points = home_score + away_score)
            validation_data: Optional (X_val, y_val) for calibration
            fit_calibrator: Whether to fit probability calibrator

        Returns:
            Self for method chaining
        """
        X_arr = self._prepare_features(X)
        y_arr = y.to_numpy() if isinstance(y, pl.Series) else np.asarray(y)

        self.logger.info(f"Training totals model on {len(y_arr)} samples")

        # Split for validation if not provided
        if validation_data is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X_arr, y_arr, test_size=0.2, random_state=42
            )
        else:
            X_train, y_train = X_arr, y_arr
            X_val = self._prepare_features(validation_data[0])
            y_val = (
                validation_data[1].to_numpy()
                if isinstance(validation_data[1], pl.Series)
                else np.asarray(validation_data[1])
            )

        # Train quantile models
        for q in self.quantiles:
            self._train_quantile_model(X_train, y_train, q)

        # Calculate residual std from median predictions
        val_preds = self._quantile_models[0.5].predict(X_val)
        residuals = y_val - val_preds
        self._residual_std = float(np.std(residuals))
        self.logger.info(f"Residual std: {self._residual_std:.2f} points")

        # Fit calibrator if requested
        if fit_calibrator and self.use_calibration:
            self._fit_calibrator(X_val, y_val)

        self.is_fitted = True

        # Create metadata
        self.metadata = ModelMetadata(
            model_type=self.MODEL_TYPE,
            model_version=self.VERSION,
            training_date=datetime.now(),
            training_seasons=[],  # Filled by caller
            n_training_samples=len(y_arr),
            feature_names=self.feature_names,
            hyperparameters={
                "quantiles": self.quantiles,
                "model_params": self.model_params,
            },
            metrics=self.evaluate(X_val, y_val),
            calibrated=self.calibrator is not None,
        )

        self.logger.info("Totals model training complete")
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
            )
            model.fit(X, y)
        elif HAS_XGB:
            # Fallback to XGBoost with quantile loss
            model = xgb.XGBRegressor(
                objective="reg:quantileerror",
                quantile_alpha=quantile,
                **{k: v for k, v in self.model_params.items()
                   if k not in ["verbose"]},
            )
            model.fit(X, y)
        else:
            # Fallback to Ridge (point estimate only)
            self.logger.warning(
                "Neither LightGBM nor XGBoost available, using Ridge regression"
            )
            model = Ridge(alpha=1.0)
            model.fit(X, y)

        self._quantile_models[quantile] = model

    def _fit_calibrator(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> None:
        """Fit probability calibrator on validation data."""
        self.logger.debug("Fitting probability calibrator...")

        # Generate predictions for various synthetic lines
        median_preds = self._quantile_models[0.5].predict(X_val)

        all_probs = []
        all_outcomes = []

        # Create calibration data by using different "lines" relative to prediction
        for offset_pct in [0.85, 0.925, 1.0, 1.075, 1.15]:
            synthetic_lines = median_preds * offset_pct
            probs = self._calculate_over_prob_from_quantiles(X_val, synthetic_lines)
            outcomes = (y_val > synthetic_lines).astype(float)
            all_probs.extend(probs)
            all_outcomes.extend(outcomes)

        self.calibrator = ProbabilityCalibrator(method="isotonic")
        self.calibrator.fit(np.array(all_probs), np.array(all_outcomes))

    def _calculate_over_prob_from_quantiles(
        self,
        X: np.ndarray,
        lines: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate over probability by interpolating between quantiles.

        Uses linear interpolation in the quantile space to estimate
        what quantile the line corresponds to, then converts to probability.
        """
        # Get all quantile predictions
        quantile_preds = {}
        for q in self.quantiles:
            quantile_preds[q] = self._quantile_models[q].predict(X)

        over_probs = np.zeros(len(X))

        for i in range(len(X)):
            line = lines[i]

            # Find where the line falls in the quantile distribution
            quantile_vals = [quantile_preds[q][i] for q in self.quantiles]

            # If line is below lowest quantile
            if line <= quantile_vals[0]:
                # Extrapolate: assume normal below
                std = (quantile_vals[-1] - quantile_vals[0]) / 2.56  # 90th - 10th span
                z = (quantile_vals[0] - line) / max(std, 0.1)
                over_probs[i] = min(0.99, self.quantiles[0] + (1 - stats.norm.cdf(-z)))
            # If line is above highest quantile
            elif line >= quantile_vals[-1]:
                std = (quantile_vals[-1] - quantile_vals[0]) / 2.56
                z = (line - quantile_vals[-1]) / max(std, 0.1)
                over_probs[i] = max(0.01, (1 - self.quantiles[-1]) * stats.norm.cdf(-z))
            else:
                # Interpolate between quantiles
                for j in range(len(self.quantiles) - 1):
                    if quantile_vals[j] <= line <= quantile_vals[j + 1]:
                        q_low, q_high = self.quantiles[j], self.quantiles[j + 1]
                        v_low, v_high = quantile_vals[j], quantile_vals[j + 1]

                        # Linear interpolation in quantile space
                        if v_high - v_low > 0:
                            interp_q = q_low + (q_high - q_low) * (line - v_low) / (v_high - v_low)
                        else:
                            interp_q = (q_low + q_high) / 2

                        over_probs[i] = 1 - interp_q
                        break

        return over_probs

    def predict(
        self,
        X: Union[pl.DataFrame, np.ndarray],
    ) -> np.ndarray:
        """
        Predict total points for multiple games.

        Args:
            X: Feature matrix

        Returns:
            Array of predicted totals (median prediction)
        """
        self._validate_fitted()
        X_arr = self._prepare_features(X)
        return self._quantile_models[0.5].predict(X_arr)

    def predict_proba(
        self,
        X: Union[pl.DataFrame, np.ndarray],
        lines: np.ndarray,
    ) -> np.ndarray:
        """
        Predict over probabilities for given lines.

        Args:
            X: Feature matrix
            lines: Vegas lines to calculate over probability for

        Returns:
            Array of over probabilities
        """
        self._validate_fitted()
        X_arr = self._prepare_features(X)

        raw_probs = self._calculate_over_prob_from_quantiles(X_arr, lines)

        # Apply calibration if available
        if self.calibrator is not None:
            return self.calibrator.calibrate(raw_probs)

        return raw_probs

    def predict_quantiles(
        self,
        X: Union[pl.DataFrame, np.ndarray],
    ) -> dict[float, np.ndarray]:
        """
        Predict all quantiles.

        Args:
            X: Feature matrix

        Returns:
            Dictionary mapping quantile to predictions
        """
        self._validate_fitted()
        X_arr = self._prepare_features(X)

        return {
            q: self._quantile_models[q].predict(X_arr)
            for q in self.quantiles
        }

    def predict_game(
        self,
        features: Union[dict, pl.DataFrame, np.ndarray],
        game_id: str,
        home_team: str,
        away_team: str,
        line: Optional[float] = None,
        over_odds: Optional[int] = None,
        under_odds: Optional[int] = None,
    ) -> TotalsPrediction:
        """
        Make a single game prediction with full context.

        Args:
            features: Feature values for the game
            game_id: Unique game identifier
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            line: Vegas total line (e.g., 47.5)
            over_odds: Over American odds (e.g., -110)
            under_odds: Under American odds (e.g., -110)

        Returns:
            TotalsPrediction with all prediction details
        """
        self._validate_fitted()

        # Prepare features
        if isinstance(features, dict):
            X = np.array([[features.get(f, 0.0) for f in self.feature_names]])
        elif isinstance(features, pl.DataFrame):
            X = features.select(self.feature_names).to_numpy()
        else:
            X = np.atleast_2d(features)

        # Get all quantile predictions
        quantile_preds = self.predict_quantiles(X)
        predicted_total = float(quantile_preds[0.5][0])

        # Estimate std from IQR
        iqr = float(quantile_preds[0.75][0] - quantile_preds[0.25][0])
        prediction_std = iqr / 1.35  # IQR to std conversion for normal

        # Calculate over/under probabilities if line provided
        over_prob = None
        under_prob = None
        over_implied_prob = None
        under_implied_prob = None
        over_edge = None
        under_edge = None

        if line is not None:
            over_prob = float(self.predict_proba(X, np.array([line]))[0])
            under_prob = 1.0 - over_prob

            if over_odds is not None and under_odds is not None:
                over_implied_prob = self._american_to_prob(over_odds)
                under_implied_prob = self._american_to_prob(under_odds)

                # Remove vig for fair comparison
                total_implied = over_implied_prob + under_implied_prob
                over_no_vig = over_implied_prob / total_implied
                under_no_vig = under_implied_prob / total_implied

                over_edge = over_prob - over_no_vig
                under_edge = under_prob - under_no_vig

        return TotalsPrediction(
            game_id=game_id,
            home_team=home_team,
            away_team=away_team,
            predicted_total=predicted_total,
            prediction_std=prediction_std,
            quantile_10=float(quantile_preds[0.10][0]),
            quantile_25=float(quantile_preds[0.25][0]),
            quantile_50=float(quantile_preds[0.50][0]),
            quantile_75=float(quantile_preds[0.75][0]),
            quantile_90=float(quantile_preds[0.90][0]),
            line=line,
            over_prob=over_prob,
            under_prob=under_prob,
            over_odds=over_odds,
            under_odds=under_odds,
            over_implied_prob=over_implied_prob,
            under_implied_prob=under_implied_prob,
            over_edge=over_edge,
            under_edge=under_edge,
            model_version=self.VERSION,
        )

    @staticmethod
    def _american_to_prob(odds: int) -> float:
        """Convert American odds to implied probability."""
        if odds > 0:
            return 100.0 / (odds + 100.0)
        else:
            return abs(odds) / (abs(odds) + 100.0)

    def get_feature_importance(self) -> dict[str, float]:
        """
        Get aggregated feature importance from median model.

        Returns importance scores from the 0.5 quantile model.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        median_model = self._quantile_models[0.5]

        if hasattr(median_model, "feature_importances_"):
            importance = median_model.feature_importances_
            return dict(sorted(
                zip(self.feature_names, importance),
                key=lambda x: x[1],
                reverse=True
            ))

        return {}

    def save(self, path: Union[str, Path]) -> None:
        """Save the complete model to disk."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        save_dict = {
            "model_type": self.MODEL_TYPE,
            "version": self.VERSION,
            "quantile_models": self._quantile_models,
            "quantiles": self.quantiles,
            "calibrator": self.calibrator,
            "residual_std": self._residual_std,
            "feature_names": self.feature_names,
            "model_params": self.model_params,
            "metadata": self.metadata,
            "is_fitted": self.is_fitted,
        }

        joblib.dump(save_dict, path)
        self.logger.info(f"Model saved to {path}")

    def load(self, path: Union[str, Path]) -> "TotalsModel":
        """Load a complete model from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        save_dict = joblib.load(path)

        if save_dict.get("model_type") != self.MODEL_TYPE:
            raise ValueError(
                f"Model type mismatch: expected {self.MODEL_TYPE}, "
                f"got {save_dict.get('model_type')}"
            )

        self._quantile_models = save_dict["quantile_models"]
        self.quantiles = save_dict["quantiles"]
        self.calibrator = save_dict["calibrator"]
        self._residual_std = save_dict["residual_std"]
        self.feature_names = save_dict["feature_names"]
        self.model_params = save_dict.get("model_params", self.MODEL_PARAMS)
        self.metadata = save_dict.get("metadata")
        self.is_fitted = save_dict.get("is_fitted", True)

        self.logger.info(f"Model loaded from {path}")
        return self
