"""
XGBoost ensemble model for NFL spread predictions.

Combines multiple models for robust predictions:
- XGBoost (primary): Gradient boosting, handles non-linear relationships
- LightGBM (secondary): Fast, handles large datasets well
- Ridge (baseline): Linear regularized regression for stability

The ensemble provides:
- Better generalization through model diversity
- Uncertainty estimates from model disagreement
- Robust predictions across different game types
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
class SpreadPrediction:
    """Container for a single spread prediction."""

    game_id: str
    home_team: str
    away_team: str

    # Point prediction
    predicted_spread: float

    # Uncertainty
    prediction_std: float
    confidence_lower: float
    confidence_upper: float

    # Cover probabilities
    home_cover_prob: float
    away_cover_prob: float

    # Betting line context
    betting_line: Optional[float] = None
    edge: Optional[float] = None

    # Metadata
    model_version: str = ""
    prediction_time: datetime = None

    def __post_init__(self):
        if self.prediction_time is None:
            self.prediction_time = datetime.now()

    @property
    def pick(self) -> str:
        """Get the model's pick (home or away)."""
        if self.betting_line is None:
            return "home" if self.predicted_spread > 0 else "away"
        return "home" if self.home_cover_prob > 0.5 else "away"

    @property
    def confidence(self) -> float:
        """Get confidence in the pick (0.5 to 1.0)."""
        return max(self.home_cover_prob, self.away_cover_prob)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "game_id": self.game_id,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "predicted_spread": self.predicted_spread,
            "prediction_std": self.prediction_std,
            "confidence_interval": [self.confidence_lower, self.confidence_upper],
            "home_cover_prob": self.home_cover_prob,
            "away_cover_prob": self.away_cover_prob,
            "betting_line": self.betting_line,
            "edge": self.edge,
            "pick": self.pick,
            "confidence": self.confidence,
            "model_version": self.model_version,
            "prediction_time": self.prediction_time.isoformat(),
        }


class SpreadModel(BaseModel):
    """
    Ensemble model for NFL game spread prediction.

    Uses weighted combination of XGBoost, LightGBM, and Ridge regression
    for robust spread predictions with uncertainty estimates.

    Example:
        >>> model = SpreadModel()
        >>> model.train(X_train, y_train)
        >>> prediction = model.predict_game(
        ...     features, game_id="2024_01_KC_BAL",
        ...     home_team="BAL", away_team="KC", line=-3.5
        ... )
        >>> print(f"Predicted: {prediction.predicted_spread:.1f}")
        >>> print(f"Home cover prob: {prediction.home_cover_prob:.1%}")
    """

    MODEL_TYPE = "spread_ensemble"
    VERSION = "1.0.0"

    # Default ensemble weights
    DEFAULT_WEIGHTS = {
        "xgb": 0.50,
        "lgb": 0.35,
        "ridge": 0.15,
    }

    # Default XGBoost hyperparameters
    XGB_PARAMS = {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
    }

    # Default LightGBM hyperparameters
    LGB_PARAMS = {
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

    # Default Ridge hyperparameters
    RIDGE_PARAMS = {
        "alpha": 1.0,
    }

    def __init__(
        self,
        ensemble_weights: Optional[dict[str, float]] = None,
        xgb_params: Optional[dict] = None,
        lgb_params: Optional[dict] = None,
        ridge_params: Optional[dict] = None,
        use_calibration: bool = True,
    ):
        """
        Initialize the spread model.

        Args:
            ensemble_weights: Weights for each model in ensemble
            xgb_params: XGBoost hyperparameters
            lgb_params: LightGBM hyperparameters
            ridge_params: Ridge regression hyperparameters
            use_calibration: Whether to calibrate probabilities
        """
        super().__init__()

        self.weights = ensemble_weights or self.DEFAULT_WEIGHTS.copy()
        self.use_calibration = use_calibration

        # Normalize weights
        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}

        # Initialize models
        self.xgb_model = None
        self.lgb_model = None
        self.ridge_model = None

        # Store hyperparameters
        self.xgb_params = {**self.XGB_PARAMS, **(xgb_params or {})}
        self.lgb_params = {**self.LGB_PARAMS, **(lgb_params or {})}
        self.ridge_params = {**self.RIDGE_PARAMS, **(ridge_params or {})}

        # Calibrator for cover probabilities
        self.calibrator: Optional[ProbabilityCalibrator] = None

        # Track prediction uncertainty
        self._residual_std: float = 10.0  # Default NFL spread std

        self.logger = logger.bind(model="spread_ensemble")

    def train(
        self,
        X: Union[pl.DataFrame, np.ndarray],
        y: Union[pl.Series, np.ndarray],
        validation_data: Optional[tuple] = None,
        early_stopping_rounds: int = 50,
        fit_calibrator: bool = True,
    ) -> "SpreadModel":
        """
        Train the ensemble model.

        Args:
            X: Feature matrix
            y: Target values (actual spreads: home_score - away_score)
            validation_data: Optional (X_val, y_val) for early stopping
            early_stopping_rounds: Rounds for early stopping
            fit_calibrator: Whether to fit probability calibrator

        Returns:
            Self for method chaining
        """
        X_arr = self._prepare_features(X)
        y_arr = y.to_numpy() if isinstance(y, pl.Series) else np.asarray(y)

        self.logger.info(f"Training spread model on {len(y_arr)} samples")

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

        # Train XGBoost
        if HAS_XGB and self.weights.get("xgb", 0) > 0:
            self._train_xgb(X_train, y_train, X_val, y_val, early_stopping_rounds)
        else:
            self.weights["xgb"] = 0
            if self.weights.get("xgb", 0) > 0:
                self.logger.warning("XGBoost not available, skipping")

        # Train LightGBM
        if HAS_LGB and self.weights.get("lgb", 0) > 0:
            self._train_lgb(X_train, y_train, X_val, y_val, early_stopping_rounds)
        else:
            self.weights["lgb"] = 0
            if self.weights.get("lgb", 0) > 0:
                self.logger.warning("LightGBM not available, skipping")

        # Train Ridge
        if self.weights.get("ridge", 0) > 0:
            self._train_ridge(X_train, y_train)

        # Re-normalize weights after potential model exclusions
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}

        # Calculate residual standard deviation for uncertainty
        val_preds = self._ensemble_predict(X_val)
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
                "weights": self.weights,
                "xgb_params": self.xgb_params,
                "lgb_params": self.lgb_params,
                "ridge_params": self.ridge_params,
            },
            metrics=self.evaluate(X_val, y_val),
            calibrated=self.calibrator is not None,
        )

        self.logger.info("Spread model training complete")
        return self

    def _train_xgb(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        early_stopping_rounds: int,
    ) -> None:
        """Train XGBoost model."""
        self.logger.debug("Training XGBoost...")

        params = self.xgb_params.copy()
        params["early_stopping_rounds"] = early_stopping_rounds

        self.xgb_model = xgb.XGBRegressor(**params)
        self.xgb_model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        best_iter = self.xgb_model.best_iteration
        self.logger.debug(f"XGBoost best iteration: {best_iter}")

    def _train_lgb(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        early_stopping_rounds: int,
    ) -> None:
        """Train LightGBM model."""
        self.logger.debug("Training LightGBM...")

        self.lgb_model = lgb.LGBMRegressor(**self.lgb_params)

        # Create callback for early stopping
        callbacks = [
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=-1),  # Disable logging
        ]

        self.lgb_model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=callbacks,
        )

        best_iter = self.lgb_model.best_iteration_
        self.logger.debug(f"LightGBM best iteration: {best_iter}")

    def _train_ridge(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> None:
        """Train Ridge regression model."""
        self.logger.debug("Training Ridge...")
        self.ridge_model = Ridge(**self.ridge_params)
        self.ridge_model.fit(X_train, y_train)

    def _fit_calibrator(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> None:
        """Fit probability calibrator on validation data."""
        self.logger.debug("Fitting probability calibrator...")

        # Generate spread predictions
        preds = self._ensemble_predict(X_val)

        # Create synthetic lines at various points for calibration
        # Use actual spread as "line" to get cover outcomes
        lines = y_val  # If predicted > actual, home covered

        # Calculate raw cover probabilities
        z_scores = (preds - lines) / max(self._residual_std, 0.1)
        raw_probs = stats.norm.cdf(z_scores)

        # Binary outcomes: did home team cover?
        covers = (y_val > lines).astype(float)  # Always 0.5 since lines = y_val

        # Create calibration data by using different "lines"
        # Add noise to create varied outcomes
        all_probs = []
        all_outcomes = []

        for offset in [-7, -3.5, 0, 3.5, 7]:
            synthetic_lines = y_val + offset
            z = (preds - synthetic_lines) / max(self._residual_std, 0.1)
            probs = stats.norm.cdf(z)
            outcomes = (y_val > synthetic_lines).astype(float)
            all_probs.extend(probs)
            all_outcomes.extend(outcomes)

        self.calibrator = ProbabilityCalibrator(method="isotonic")
        self.calibrator.fit(np.array(all_probs), np.array(all_outcomes))

    def _ensemble_predict(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble predictions."""
        predictions = np.zeros(len(X))

        if self.xgb_model is not None and self.weights.get("xgb", 0) > 0:
            predictions += self.weights["xgb"] * self.xgb_model.predict(X)

        if self.lgb_model is not None and self.weights.get("lgb", 0) > 0:
            predictions += self.weights["lgb"] * self.lgb_model.predict(X)

        if self.ridge_model is not None and self.weights.get("ridge", 0) > 0:
            predictions += self.weights["ridge"] * self.ridge_model.predict(X)

        return predictions

    def _get_model_predictions(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Get individual model predictions for uncertainty estimation."""
        preds = {}

        if self.xgb_model is not None:
            preds["xgb"] = self.xgb_model.predict(X)

        if self.lgb_model is not None:
            preds["lgb"] = self.lgb_model.predict(X)

        if self.ridge_model is not None:
            preds["ridge"] = self.ridge_model.predict(X)

        return preds

    def predict(
        self,
        X: Union[pl.DataFrame, np.ndarray],
    ) -> np.ndarray:
        """
        Predict spreads for multiple games.

        Args:
            X: Feature matrix

        Returns:
            Array of predicted spreads (positive = home team favored)
        """
        self._validate_fitted()
        X_arr = self._prepare_features(X)
        return self._ensemble_predict(X_arr)

    def predict_proba(
        self,
        X: Union[pl.DataFrame, np.ndarray],
        lines: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Predict home team cover probabilities.

        Args:
            X: Feature matrix
            lines: Betting lines (if None, uses predicted spreads)

        Returns:
            Array of home cover probabilities
        """
        self._validate_fitted()
        X_arr = self._prepare_features(X)

        predictions = self._ensemble_predict(X_arr)

        if lines is None:
            lines = predictions  # 50% cover prob when line = prediction

        # Calculate raw probabilities
        z_scores = (predictions - lines) / max(self._residual_std, 0.1)
        raw_probs = stats.norm.cdf(z_scores)

        # Apply calibration if available
        if self.calibrator is not None:
            return self.calibrator.calibrate(raw_probs)

        return raw_probs

    def predict_with_confidence(
        self,
        X: Union[pl.DataFrame, np.ndarray],
        confidence_level: float = 0.90,
    ) -> PredictionResult:
        """
        Predict spreads with confidence intervals.

        Args:
            X: Feature matrix
            confidence_level: Confidence level (default 0.90)

        Returns:
            PredictionResult with predictions and uncertainty
        """
        self._validate_fitted()
        X_arr = self._prepare_features(X)

        # Get ensemble prediction
        predictions = self._ensemble_predict(X_arr)

        # Get individual model predictions for uncertainty
        model_preds = self._get_model_predictions(X_arr)

        # Estimate uncertainty from model disagreement + residual std
        if len(model_preds) > 1:
            preds_array = np.array(list(model_preds.values()))
            model_std = np.std(preds_array, axis=0)
            total_std = np.sqrt(model_std**2 + self._residual_std**2)
        else:
            total_std = np.full(len(predictions), self._residual_std)

        # Calculate confidence intervals
        z = stats.norm.ppf((1 + confidence_level) / 2)
        lower = predictions - z * total_std
        upper = predictions + z * total_std

        return PredictionResult(
            prediction=predictions,
            confidence_lower=lower,
            confidence_upper=upper,
            std=total_std,
            model_version=self.VERSION,
            feature_names=self.feature_names,
        )

    def predict_game(
        self,
        features: Union[dict, pl.DataFrame, np.ndarray],
        game_id: str,
        home_team: str,
        away_team: str,
        line: Optional[float] = None,
    ) -> SpreadPrediction:
        """
        Make a single game prediction with full context.

        Args:
            features: Feature values for the game
            game_id: Unique game identifier
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            line: Optional betting line

        Returns:
            SpreadPrediction with all prediction details
        """
        self._validate_fitted()

        # Prepare features
        if isinstance(features, dict):
            X = np.array([[features.get(f, 0.0) for f in self.feature_names]])
        elif isinstance(features, pl.DataFrame):
            X = features.select(self.feature_names).to_numpy()
        else:
            X = np.atleast_2d(features)

        # Get prediction with uncertainty
        result = self.predict_with_confidence(X)
        predicted_spread = float(result.prediction[0])
        pred_std = float(result.std[0])

        # Calculate cover probabilities
        if line is not None:
            home_cover_prob = float(self.predict_proba(X, np.array([line]))[0])
        else:
            home_cover_prob = 0.5  # No line, no edge

        # Calculate edge if line provided
        edge = None
        if line is not None:
            # Edge = predicted prob - implied prob (50% at fair line)
            edge = home_cover_prob - 0.5

        return SpreadPrediction(
            game_id=game_id,
            home_team=home_team,
            away_team=away_team,
            predicted_spread=predicted_spread,
            prediction_std=pred_std,
            confidence_lower=float(result.confidence_lower[0]),
            confidence_upper=float(result.confidence_upper[0]),
            home_cover_prob=home_cover_prob,
            away_cover_prob=1 - home_cover_prob,
            betting_line=line,
            edge=edge,
            model_version=self.VERSION,
        )

    def get_feature_importance(self) -> dict[str, float]:
        """
        Get aggregated feature importance from ensemble.

        Returns weighted average of importance scores from all models.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        importance = {}

        # XGBoost importance
        if self.xgb_model is not None and self.weights.get("xgb", 0) > 0:
            xgb_imp = self.xgb_model.feature_importances_
            for i, name in enumerate(self.feature_names):
                importance[name] = importance.get(name, 0) + self.weights["xgb"] * xgb_imp[i]

        # LightGBM importance
        if self.lgb_model is not None and self.weights.get("lgb", 0) > 0:
            lgb_imp = self.lgb_model.feature_importances_
            for i, name in enumerate(self.feature_names):
                importance[name] = importance.get(name, 0) + self.weights["lgb"] * lgb_imp[i]

        # Ridge importance (use absolute coefficients)
        if self.ridge_model is not None and self.weights.get("ridge", 0) > 0:
            ridge_imp = np.abs(self.ridge_model.coef_)
            ridge_imp = ridge_imp / np.sum(ridge_imp)  # Normalize
            for i, name in enumerate(self.feature_names):
                importance[name] = importance.get(name, 0) + self.weights["ridge"] * ridge_imp[i]

        # Sort by importance
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    def save(self, path: Union[str, Path]) -> None:
        """Save the complete model to disk."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        save_dict = {
            "model_type": self.MODEL_TYPE,
            "version": self.VERSION,
            "weights": self.weights,
            "xgb_model": self.xgb_model,
            "lgb_model": self.lgb_model,
            "ridge_model": self.ridge_model,
            "calibrator": self.calibrator,
            "residual_std": self._residual_std,
            "feature_names": self.feature_names,
            "metadata": self.metadata,
            "is_fitted": self.is_fitted,
            "xgb_params": self.xgb_params,
            "lgb_params": self.lgb_params,
            "ridge_params": self.ridge_params,
        }

        joblib.dump(save_dict, path)
        self.logger.info(f"Model saved to {path}")

    def load(self, path: Union[str, Path]) -> "SpreadModel":
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

        self.weights = save_dict["weights"]
        self.xgb_model = save_dict["xgb_model"]
        self.lgb_model = save_dict["lgb_model"]
        self.ridge_model = save_dict["ridge_model"]
        self.calibrator = save_dict["calibrator"]
        self._residual_std = save_dict["residual_std"]
        self.feature_names = save_dict["feature_names"]
        self.metadata = save_dict.get("metadata")
        self.is_fitted = save_dict.get("is_fitted", True)
        self.xgb_params = save_dict.get("xgb_params", self.XGB_PARAMS)
        self.lgb_params = save_dict.get("lgb_params", self.LGB_PARAMS)
        self.ridge_params = save_dict.get("ridge_params", self.RIDGE_PARAMS)

        self.logger.info(f"Model loaded from {path}")
        return self
