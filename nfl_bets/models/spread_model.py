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
- Optuna hyperparameter optimization for best performance
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
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler

# NFL Key Numbers for spread betting
# Games frequently land on these margins, affecting cover probability
NFL_KEY_NUMBERS = {
    3: 0.15,   # Field goal - most common margin (~15% of games)
    7: 0.10,   # TD - second most common (~10% of games)
    10: 0.05,  # TD + FG
    6: 0.04,   # Two FGs
    14: 0.04,  # Two TDs
    4: 0.03,   # FG + safety or 2-pt
    17: 0.03,  # TD + TD + FG
}

try:
    import optuna
    from optuna.samplers import TPESampler
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    optuna = None

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

    # Default XGBoost hyperparameters (stronger regularization to prevent overfitting)
    XGB_PARAMS = {
        "n_estimators": 300,          # Reduced from 500
        "max_depth": 4,               # Reduced from 6 (shallower trees)
        "learning_rate": 0.03,        # Reduced from 0.05 (slower learning)
        "subsample": 0.7,             # Reduced from 0.8 (more regularization)
        "colsample_bytree": 0.6,      # Reduced from 0.8 (more regularization)
        "min_child_weight": 10,       # Increased from 3 (larger leaves)
        "reg_alpha": 1.0,             # Increased from 0.1 (L1 regularization)
        "reg_lambda": 5.0,            # Increased from 1.0 (L2 regularization)
        "gamma": 0.1,                 # Added: min loss reduction for splits
        "random_state": 42,
        "n_jobs": -1,
    }

    # Default LightGBM hyperparameters (stronger regularization)
    LGB_PARAMS = {
        "n_estimators": 300,          # Reduced from 500
        "max_depth": 4,               # Reduced from 6
        "learning_rate": 0.03,        # Reduced from 0.05
        "subsample": 0.7,             # Reduced from 0.8
        "colsample_bytree": 0.6,      # Reduced from 0.8
        "min_child_samples": 50,      # Increased from 20 (larger leaves)
        "reg_alpha": 1.0,             # Increased from 0.1
        "reg_lambda": 5.0,            # Increased from 1.0
        "min_split_gain": 0.1,        # Added: min gain for splits
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    # Default Ridge hyperparameters (stronger regularization)
    RIDGE_PARAMS = {
        "alpha": 10.0,                # Increased from 1.0
    }

    def __init__(
        self,
        ensemble_weights: Optional[dict[str, float]] = None,
        xgb_params: Optional[dict] = None,
        lgb_params: Optional[dict] = None,
        ridge_params: Optional[dict] = None,
        use_calibration: bool = True,
        use_feature_selection: bool = True,
        max_features: int = 50,
    ):
        """
        Initialize the spread model.

        Args:
            ensemble_weights: Weights for each model in ensemble
            xgb_params: XGBoost hyperparameters
            lgb_params: LightGBM hyperparameters
            ridge_params: Ridge regression hyperparameters
            use_calibration: Whether to calibrate probabilities
            use_feature_selection: Whether to apply feature selection
            max_features: Maximum number of features to keep after selection
        """
        super().__init__()

        self.weights = ensemble_weights or self.DEFAULT_WEIGHTS.copy()
        self.use_calibration = use_calibration
        self.use_feature_selection = use_feature_selection
        self.max_features = max_features

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

        # Feature selection components
        self.variance_selector: Optional[VarianceThreshold] = None
        self.k_best_selector: Optional[SelectKBest] = None
        self.scaler: Optional[StandardScaler] = None
        self.selected_feature_indices: Optional[np.ndarray] = None
        self.selected_feature_names: Optional[list[str]] = None

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
        season_week_index: Optional[np.ndarray] = None,
    ) -> "SpreadModel":
        """
        Train the ensemble model using proper time-series split.

        CRITICAL: Uses chronological split to prevent data leakage.
        Data is sorted by time and the last 20% is used for validation.
        NO shuffling or stratification is used.

        Args:
            X: Feature matrix (should be ordered chronologically)
            y: Target values (actual spreads: home_score - away_score)
            validation_data: Optional (X_val, y_val) for early stopping
            early_stopping_rounds: Rounds for early stopping
            fit_calibrator: Whether to fit probability calibrator
            season_week_index: Optional array of (season * 100 + week) for sorting

        Returns:
            Self for method chaining
        """
        X_arr = self._prepare_features(X)
        y_arr = y.to_numpy() if isinstance(y, pl.Series) else np.asarray(y)

        self.logger.info(f"Training spread model on {len(y_arr)} samples, {X_arr.shape[1]} features")

        # If season_week_index provided, sort chronologically
        if season_week_index is not None:
            sort_idx = np.argsort(season_week_index)
            X_arr = X_arr[sort_idx]
            y_arr = y_arr[sort_idx]
            self.logger.info("Data sorted chronologically by season/week")

        # Apply feature selection if enabled
        if self.use_feature_selection:
            X_arr = self._fit_feature_selection(X_arr, y_arr)
            self.logger.info(f"After feature selection: {X_arr.shape[1]} features retained")

        # TIME-SERIES SPLIT: Use last 20% chronologically for validation
        # CRITICAL: No shuffling, no stratification - pure temporal split
        if validation_data is None:
            n_samples = len(y_arr)
            split_idx = int(n_samples * 0.8)  # 80% train, 20% validation

            # Strictly chronological split - NO randomness
            X_train = X_arr[:split_idx]
            y_train = y_arr[:split_idx]
            X_val = X_arr[split_idx:]
            y_val = y_arr[split_idx:]

            self.logger.info(
                f"Time-series split: {len(y_train)} train / {len(y_val)} validation "
                f"(last {100 * len(y_val) / n_samples:.1f}% for validation)"
            )
        else:
            X_train, y_train = X_arr, y_arr
            X_val_raw = self._prepare_features(validation_data[0])
            # Apply same feature selection to validation data
            if self.use_feature_selection:
                X_val = self._apply_feature_selection(X_val_raw)
            else:
                X_val = X_val_raw
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

    def _fit_feature_selection(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        """
        Fit and apply feature selection pipeline.

        Steps:
        1. Remove zero-variance features (useless for prediction)
        2. Select top K features by F-regression score
        3. Track selected feature indices for prediction time

        Args:
            X: Feature matrix
            y: Target values

        Returns:
            Reduced feature matrix
        """
        original_n_features = X.shape[1]

        # Handle NaN/Inf values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Step 1: Remove zero-variance features
        self.variance_selector = VarianceThreshold(threshold=1e-10)
        try:
            X_var = self.variance_selector.fit_transform(X)
            variance_mask = self.variance_selector.get_support()
        except Exception as e:
            self.logger.warning(f"Variance threshold failed: {e}, skipping")
            X_var = X
            variance_mask = np.ones(X.shape[1], dtype=bool)

        self.logger.debug(
            f"Variance filter: {original_n_features} -> {X_var.shape[1]} features"
        )

        # Step 2: Select top K features by F-regression score
        k = min(self.max_features, X_var.shape[1])
        if k < X_var.shape[1]:
            self.k_best_selector = SelectKBest(f_regression, k=k)
            try:
                X_selected = self.k_best_selector.fit_transform(X_var, y)
                k_best_mask = self.k_best_selector.get_support()
            except Exception as e:
                self.logger.warning(f"SelectKBest failed: {e}, skipping")
                X_selected = X_var
                k_best_mask = np.ones(X_var.shape[1], dtype=bool)
        else:
            X_selected = X_var
            k_best_mask = np.ones(X_var.shape[1], dtype=bool)

        self.logger.debug(
            f"K-best selection: {X_var.shape[1]} -> {X_selected.shape[1]} features"
        )

        # Track which original features were selected
        variance_indices = np.where(variance_mask)[0]
        k_best_indices = np.where(k_best_mask)[0]
        self.selected_feature_indices = variance_indices[k_best_indices]

        # Update selected feature names if available
        if self.feature_names is not None and len(self.feature_names) > 0:
            all_names = list(self.feature_names)
            self.selected_feature_names = [
                all_names[i] for i in self.selected_feature_indices
            ]
            self.logger.debug(f"Selected features: {self.selected_feature_names[:10]}...")

        return X_selected

    def _apply_feature_selection(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Apply fitted feature selection to new data.

        Args:
            X: Feature matrix

        Returns:
            Reduced feature matrix
        """
        # Handle NaN/Inf values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Apply variance selector
        if self.variance_selector is not None:
            try:
                X = self.variance_selector.transform(X)
            except Exception as e:
                self.logger.warning(f"Variance transform failed: {e}")

        # Apply k-best selector
        if self.k_best_selector is not None:
            try:
                X = self.k_best_selector.transform(X)
            except Exception as e:
                self.logger.warning(f"K-best transform failed: {e}")

        return X

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
        t_df: int = 7,  # Use same t-distribution as predict_proba
    ) -> None:
        """
        Fit probability calibrator on validation data.

        Uses t-distribution (df=7) to match predict_proba implementation.
        """
        self.logger.debug("Fitting probability calibrator...")

        # Generate spread predictions
        preds = self._ensemble_predict(X_val)
        residual_std = max(self._residual_std, 0.1)

        # Create calibration data by using different synthetic "lines"
        # This generates varied outcomes for calibration training
        all_probs = []
        all_outcomes = []

        # Use key numbers as offsets for more realistic calibration
        for offset in [-7, -3.5, -3, 0, 3, 3.5, 7]:
            synthetic_lines = y_val + offset
            scaled_diff = (preds - synthetic_lines) / residual_std

            # Use t-distribution (same as predict_proba)
            probs = stats.t.cdf(scaled_diff, df=t_df)

            # Binary outcomes: did home team cover the synthetic line?
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
        # Apply feature selection if enabled
        if self.use_feature_selection and self.variance_selector is not None:
            X_arr = self._apply_feature_selection(X_arr)
        return self._ensemble_predict(X_arr)

    def predict_proba(
        self,
        X: Union[pl.DataFrame, np.ndarray],
        lines: Optional[np.ndarray] = None,
        use_key_number_adjustment: bool = True,
        use_t_distribution: bool = True,
        t_df: int = 7,  # Degrees of freedom for t-distribution
    ) -> np.ndarray:
        """
        Predict home team cover probabilities with key number adjustment.

        CRITICAL IMPROVEMENTS:
        1. Uses t-distribution (df=7) instead of normal distribution
           - Heavier tails better capture NFL score variance
           - Normal underestimates probability of extreme outcomes

        2. Key number adjustment for lines near 3 or 7
           - Games land on 3 (field goal) ~15% of the time
           - Games land on 7 (touchdown) ~10% of the time
           - Crossing these numbers significantly changes cover probability

        Args:
            X: Feature matrix
            lines: Betting lines (if None, uses predicted spreads)
            use_key_number_adjustment: Apply adjustment for key numbers
            use_t_distribution: Use t-distribution instead of normal
            t_df: Degrees of freedom for t-distribution (lower = heavier tails)

        Returns:
            Array of home cover probabilities
        """
        self._validate_fitted()
        X_arr = self._prepare_features(X)
        # Apply feature selection if enabled
        if self.use_feature_selection and self.variance_selector is not None:
            X_arr = self._apply_feature_selection(X_arr)

        predictions = self._ensemble_predict(X_arr)

        if lines is None:
            lines = predictions  # 50% cover prob when line = prediction

        # Ensure lines is an array
        lines = np.atleast_1d(lines)

        # Calculate scaled differences
        residual_std = max(self._residual_std, 0.1)
        scaled_diff = (predictions - lines) / residual_std

        # Use t-distribution for heavier tails (better for NFL)
        # NFL score distributions have more variance than normal suggests
        if use_t_distribution:
            raw_probs = stats.t.cdf(scaled_diff, df=t_df)
        else:
            raw_probs = stats.norm.cdf(scaled_diff)

        # Apply key number adjustment
        if use_key_number_adjustment:
            raw_probs = self._apply_key_number_adjustment(
                raw_probs, predictions, lines, residual_std
            )

        # Apply calibration if available
        if self.calibrator is not None:
            return self.calibrator.calibrate(raw_probs)

        return raw_probs

    def _apply_key_number_adjustment(
        self,
        probs: np.ndarray,
        predictions: np.ndarray,
        lines: np.ndarray,
        residual_std: float,
    ) -> np.ndarray:
        """
        Adjust cover probabilities for NFL key numbers.

        Key numbers (3, 7, etc.) represent common final margins.
        When lines are near these numbers, cover probability
        changes non-linearly because:
        - If line is 2.5, ~15% of outcomes land exactly on 3
        - If line is 3.5, those 3-point wins become losses
        - This creates a ~7.5% swing in cover probability

        Args:
            probs: Raw cover probabilities
            predictions: Model predicted spreads
            lines: Betting lines
            residual_std: Model residual standard deviation

        Returns:
            Adjusted probabilities
        """
        adjusted_probs = probs.copy()

        for i, (pred, line) in enumerate(zip(predictions, lines)):
            # Check proximity to each key number
            for key_num, frequency in NFL_KEY_NUMBERS.items():
                # Calculate adjustment based on whether line crosses key number
                # relative to our prediction

                # Distance from prediction to key number
                pred_to_key = pred - key_num

                # Check if line is on opposite side of key number from prediction
                # This is when key number adjustment matters most
                line_to_key = line - key_num

                # If prediction and line are on opposite sides of key number
                # AND we're close to the key number, adjust probability
                if np.sign(pred_to_key) != np.sign(line_to_key):
                    # How close is the line to the key number?
                    key_proximity = abs(line_to_key)

                    if key_proximity <= 1.0:
                        # Adjustment factor: higher when closer to key number
                        # Max adjustment is half the frequency
                        # (since key number could go either way)
                        adjustment_factor = frequency * 0.5 * (1.0 - key_proximity)

                        # If prediction suggests we're on winning side of key,
                        # we get extra probability from games landing on key
                        if pred > line:
                            adjusted_probs[i] += adjustment_factor
                        else:
                            adjusted_probs[i] -= adjustment_factor

                # Also check for negative key numbers (other team winning)
                pred_to_neg_key = pred - (-key_num)
                line_to_neg_key = line - (-key_num)

                if np.sign(pred_to_neg_key) != np.sign(line_to_neg_key):
                    key_proximity = abs(line_to_neg_key)

                    if key_proximity <= 1.0:
                        adjustment_factor = frequency * 0.5 * (1.0 - key_proximity)

                        if pred > line:
                            adjusted_probs[i] += adjustment_factor
                        else:
                            adjusted_probs[i] -= adjustment_factor

        # Clip to valid probability range
        adjusted_probs = np.clip(adjusted_probs, 0.01, 0.99)

        return adjusted_probs

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
        # Apply feature selection if enabled
        if self.use_feature_selection and self.variance_selector is not None:
            X_arr = self._apply_feature_selection(X_arr)

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

    @classmethod
    def optimize_hyperparameters(
        cls,
        X: Union[pl.DataFrame, np.ndarray],
        y: Union[pl.Series, np.ndarray],
        n_trials: int = 50,
        n_cv_splits: int = 5,
        timeout: Optional[int] = None,
        study_name: str = "spread_model_optimization",
    ) -> tuple["SpreadModel", dict[str, Any]]:
        """
        Optimize hyperparameters using Optuna with time-series cross-validation.

        Uses TPE (Tree-structured Parzen Estimator) sampler for efficient
        hyperparameter search. Time-series CV ensures no look-ahead bias.

        Args:
            X: Feature matrix
            y: Target values
            n_trials: Number of optimization trials
            n_cv_splits: Number of time-series CV splits
            timeout: Optional timeout in seconds
            study_name: Name for the Optuna study

        Returns:
            Tuple of (optimized model, best hyperparameters)
        """
        if not HAS_OPTUNA:
            raise ImportError("Optuna is required for hyperparameter optimization. pip install optuna")

        # Prepare data
        if isinstance(X, pl.DataFrame):
            X_arr = X.to_numpy()
            feature_names = X.columns
        else:
            X_arr = np.asarray(X)
            feature_names = None

        y_arr = y.to_numpy() if isinstance(y, pl.Series) else np.asarray(y)

        logger.info(f"Starting Optuna optimization with {n_trials} trials")

        def objective(trial: optuna.Trial) -> float:
            """Objective function for Optuna optimization."""
            # XGBoost hyperparameters
            xgb_params = {
                "n_estimators": trial.suggest_int("xgb_n_estimators", 100, 1000),
                "max_depth": trial.suggest_int("xgb_max_depth", 3, 10),
                "learning_rate": trial.suggest_float("xgb_learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("xgb_subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("xgb_colsample_bytree", 0.5, 1.0),
                "min_child_weight": trial.suggest_int("xgb_min_child_weight", 1, 10),
                "reg_alpha": trial.suggest_float("xgb_reg_alpha", 1e-3, 10.0, log=True),
                "reg_lambda": trial.suggest_float("xgb_reg_lambda", 1e-3, 10.0, log=True),
                "random_state": 42,
                "n_jobs": -1,
            }

            # LightGBM hyperparameters
            lgb_params = {
                "n_estimators": trial.suggest_int("lgb_n_estimators", 100, 1000),
                "max_depth": trial.suggest_int("lgb_max_depth", 3, 10),
                "learning_rate": trial.suggest_float("lgb_learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("lgb_subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("lgb_colsample_bytree", 0.5, 1.0),
                "min_child_samples": trial.suggest_int("lgb_min_child_samples", 5, 50),
                "reg_alpha": trial.suggest_float("lgb_reg_alpha", 1e-3, 10.0, log=True),
                "reg_lambda": trial.suggest_float("lgb_reg_lambda", 1e-3, 10.0, log=True),
                "random_state": 42,
                "n_jobs": -1,
                "verbose": -1,
            }

            # Ridge hyperparameters
            ridge_params = {
                "alpha": trial.suggest_float("ridge_alpha", 0.01, 100.0, log=True),
            }

            # Ensemble weights
            xgb_weight = trial.suggest_float("xgb_weight", 0.2, 0.7)
            lgb_weight = trial.suggest_float("lgb_weight", 0.1, 0.5)
            ridge_weight = 1.0 - xgb_weight - lgb_weight
            if ridge_weight < 0.05:
                return float("inf")  # Invalid weight combination

            weights = {
                "xgb": xgb_weight,
                "lgb": lgb_weight,
                "ridge": ridge_weight,
            }

            # Time-series cross-validation
            tscv = TimeSeriesSplit(n_splits=n_cv_splits)
            cv_scores = []

            for train_idx, val_idx in tscv.split(X_arr):
                X_train, X_val = X_arr[train_idx], X_arr[val_idx]
                y_train, y_val = y_arr[train_idx], y_arr[val_idx]

                # Create and train model
                model = cls(
                    ensemble_weights=weights,
                    xgb_params=xgb_params,
                    lgb_params=lgb_params,
                    ridge_params=ridge_params,
                    use_calibration=False,  # Skip calibration during CV
                )

                try:
                    model.train(
                        X_train, y_train,
                        validation_data=(X_val, y_val),
                        early_stopping_rounds=30,
                        fit_calibrator=False,
                    )

                    # Calculate MAE on validation
                    preds = model.predict(X_val)
                    mae = np.mean(np.abs(y_val - preds))
                    cv_scores.append(mae)

                except Exception as e:
                    logger.warning(f"Trial failed: {e}")
                    return float("inf")

            mean_mae = np.mean(cv_scores)
            trial.set_user_attr("cv_scores", cv_scores)

            return mean_mae

        # Create study with TPE sampler
        sampler = TPESampler(seed=42)
        study = optuna.create_study(
            study_name=study_name,
            direction="minimize",
            sampler=sampler,
        )

        # Optimize
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True,
        )

        # Extract best parameters
        best_params = study.best_params
        best_mae = study.best_value

        logger.info(f"Best MAE: {best_mae:.3f}")
        logger.info(f"Best params: {best_params}")

        # Build final model with best parameters
        xgb_params = {
            "n_estimators": best_params["xgb_n_estimators"],
            "max_depth": best_params["xgb_max_depth"],
            "learning_rate": best_params["xgb_learning_rate"],
            "subsample": best_params["xgb_subsample"],
            "colsample_bytree": best_params["xgb_colsample_bytree"],
            "min_child_weight": best_params["xgb_min_child_weight"],
            "reg_alpha": best_params["xgb_reg_alpha"],
            "reg_lambda": best_params["xgb_reg_lambda"],
            "random_state": 42,
            "n_jobs": -1,
        }

        lgb_params = {
            "n_estimators": best_params["lgb_n_estimators"],
            "max_depth": best_params["lgb_max_depth"],
            "learning_rate": best_params["lgb_learning_rate"],
            "subsample": best_params["lgb_subsample"],
            "colsample_bytree": best_params["lgb_colsample_bytree"],
            "min_child_samples": best_params["lgb_min_child_samples"],
            "reg_alpha": best_params["lgb_reg_alpha"],
            "reg_lambda": best_params["lgb_reg_lambda"],
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }

        ridge_params = {
            "alpha": best_params["ridge_alpha"],
        }

        weights = {
            "xgb": best_params["xgb_weight"],
            "lgb": best_params["lgb_weight"],
            "ridge": 1.0 - best_params["xgb_weight"] - best_params["lgb_weight"],
        }

        # Train final model on all data
        final_model = cls(
            ensemble_weights=weights,
            xgb_params=xgb_params,
            lgb_params=lgb_params,
            ridge_params=ridge_params,
            use_calibration=True,
        )

        # CRITICAL: Use time-series split for final model (last 20% chronologically)
        # NO random splitting - data should already be ordered by time
        n_samples = len(y_arr)
        split_idx = int(n_samples * 0.8)
        X_train, X_val = X_arr[:split_idx], X_arr[split_idx:]
        y_train, y_val = y_arr[:split_idx], y_arr[split_idx:]

        final_model.train(
            X_train, y_train,
            validation_data=(X_val, y_val),
            early_stopping_rounds=50,
            fit_calibrator=True,
        )

        if feature_names is not None:
            final_model.feature_names = list(feature_names)

        return final_model, {
            "best_params": best_params,
            "best_mae": best_mae,
            "n_trials": n_trials,
            "study_name": study_name,
        }

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
