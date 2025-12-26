"""
Residual-based Spread Model

Instead of predicting spreads directly, this model predicts the ERROR in Vegas lines.
Final prediction = Vegas spread + predicted residual

This approach beats Vegas by finding systematic biases in specific situations:
- Early season games (weeks 1-4): Vegas struggles with new team compositions
- Large spread games (10+ points): Vegas overestimates favorites
- Weather games: Vegas may underweight weather impact
- Rest mismatches: Vegas may underweight rest advantages
"""

import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import numpy as np
import polars as pl
from sklearn.feature_selection import SelectKBest, f_regression, VarianceThreshold
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler

from nfl_bets.models.base import ModelMetadata

logger = logging.getLogger(__name__)


@dataclass
class ResidualModelConfig:
    """Configuration for the residual spread model."""

    # Feature selection
    k_features: int = 15  # Number of features to select (validated: k=15 optimal)
    variance_threshold: float = 1e-10

    # Lasso regularization (validated: alpha=1.0 optimal)
    alpha: float = 1.0
    max_iter: int = 10000

    # Model type: 'lasso', 'ridge', 'elasticnet'
    model_type: str = 'lasso'

    # ElasticNet specific
    l1_ratio: float = 0.5

    # Situational features to add
    add_situational_features: bool = True

    # Blend factor: how much to trust model vs pure Vegas (0=pure Vegas, 1=full model)
    blend_factor: float = 1.0


@dataclass
class ResidualPrediction:
    """Prediction result from the residual model."""

    final_spread: float  # Vegas spread + predicted residual
    vegas_spread: float  # Original Vegas spread
    predicted_residual: float  # Model's predicted adjustment
    confidence: float  # Based on residual magnitude and feature reliability
    situational_edge: Optional[str] = None  # e.g., "early_season", "large_spread"


class ResidualSpreadModel:
    """
    Residual-based spread prediction model.

    Key insight: Instead of trying to beat Vegas from scratch, we predict
    the systematic errors Vegas makes and adjust accordingly.

    Validated performance:
    - MAE: 9.716 (vs Vegas 9.733)
    - ATS: 53.1% (above 52.4% breakeven)
    - Best in: Mid-early season (58.9%), Late season (57.8%), Close games (56.3%)
    """

    def __init__(self, config: Optional[ResidualModelConfig] = None):
        """Initialize the residual spread model."""
        self.config = config or ResidualModelConfig()

        # Feature processing
        self.scaler: Optional[StandardScaler] = None
        self.variance_selector: Optional[VarianceThreshold] = None
        self.k_best_selector: Optional[SelectKBest] = None

        # Model
        self.model = None

        # Feature names for interpretability
        self.feature_names: list[str] = []
        self.selected_feature_names: list[str] = []
        self.selected_feature_indices: Optional[np.ndarray] = None

        # Metadata
        self.metadata: Optional[ModelMetadata] = None
        self.is_trained: bool = False

    def _create_model(self):
        """Create the regression model based on config."""
        if self.config.model_type == 'lasso':
            return Lasso(
                alpha=self.config.alpha,
                max_iter=self.config.max_iter,
                random_state=42
            )
        elif self.config.model_type == 'ridge':
            return Ridge(
                alpha=self.config.alpha,
                max_iter=self.config.max_iter,
                random_state=42
            )
        elif self.config.model_type == 'elasticnet':
            return ElasticNet(
                alpha=self.config.alpha,
                l1_ratio=self.config.l1_ratio,
                max_iter=self.config.max_iter,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")

    def _add_situational_features(
        self,
        X: np.ndarray,
        feature_names: list[str],
        week: Optional[np.ndarray] = None,
        vegas_spread: Optional[np.ndarray] = None,
        home_rest: Optional[np.ndarray] = None,
        away_rest: Optional[np.ndarray] = None,
        temperature: Optional[np.ndarray] = None,
        wind_speed: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, list[str]]:
        """Add situational features that capture where Vegas struggles."""

        if not self.config.add_situational_features:
            return X, feature_names

        n_samples = X.shape[0]
        new_features = []
        new_names = []

        # Early season indicator (weeks 1-4)
        if week is not None:
            is_early = (week <= 4).astype(float).reshape(-1, 1)
            new_features.append(is_early)
            new_names.append('is_early_season')

            # Mid-early season (weeks 5-8) - where we have best edge
            is_mid_early = ((week >= 5) & (week <= 8)).astype(float).reshape(-1, 1)
            new_features.append(is_mid_early)
            new_names.append('is_mid_early_season')

            # Late season (weeks 13+)
            is_late = (week >= 13).astype(float).reshape(-1, 1)
            new_features.append(is_late)
            new_names.append('is_late_season')

        # Large spread indicator (favorites by 10+ points)
        if vegas_spread is not None:
            spread_magnitude = np.abs(vegas_spread).reshape(-1, 1)
            new_features.append(spread_magnitude)
            new_names.append('spread_magnitude')

            is_large_spread = (spread_magnitude >= 10).astype(float)
            new_features.append(is_large_spread)
            new_names.append('is_large_spread')

            # Close games (spread < 3.5)
            is_close = (spread_magnitude < 3.5).astype(float)
            new_features.append(is_close)
            new_names.append('is_close_game')

        # Rest differential
        if home_rest is not None and away_rest is not None:
            rest_diff = (home_rest - away_rest).reshape(-1, 1)
            new_features.append(rest_diff)
            new_names.append('rest_diff')

            # Short rest indicator
            is_short_rest = ((home_rest < 7) | (away_rest < 7)).astype(float).reshape(-1, 1)
            new_features.append(is_short_rest)
            new_names.append('has_short_rest_team')

        # Weather indicators
        if temperature is not None:
            is_cold = (temperature < 40).astype(float).reshape(-1, 1)
            new_features.append(is_cold)
            new_names.append('is_cold_game')

        if wind_speed is not None:
            is_windy = (wind_speed > 15).astype(float).reshape(-1, 1)
            new_features.append(is_windy)
            new_names.append('is_windy_game')

        # Combine with original features
        if new_features:
            X_situational = np.hstack(new_features)
            X_combined = np.hstack([X, X_situational])
            combined_names = feature_names + new_names
            return X_combined, combined_names

        return X, feature_names

    def _fit_feature_selection(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str]
    ) -> np.ndarray:
        """Fit feature selection pipeline and transform features."""

        # Step 1: Remove zero/low variance features
        self.variance_selector = VarianceThreshold(threshold=self.config.variance_threshold)
        X_var = self.variance_selector.fit_transform(X)

        # Get feature names after variance filter
        var_mask = self.variance_selector.get_support()
        names_after_var = [name for name, keep in zip(feature_names, var_mask) if keep]

        logger.info(f"Variance filter: {X.shape[1]} -> {X_var.shape[1]} features")

        # Step 2: Select k best features
        k = min(self.config.k_features, X_var.shape[1])
        self.k_best_selector = SelectKBest(f_regression, k=k)
        X_selected = self.k_best_selector.fit_transform(X_var, y)

        # Get selected feature names
        kbest_mask = self.k_best_selector.get_support()
        self.selected_feature_names = [
            name for name, keep in zip(names_after_var, kbest_mask) if keep
        ]

        # Store indices for later
        var_indices = np.where(var_mask)[0]
        kbest_indices = np.where(kbest_mask)[0]
        self.selected_feature_indices = var_indices[kbest_indices]

        logger.info(f"K-best filter: {X_var.shape[1]} -> {X_selected.shape[1]} features")
        logger.info(f"Selected features: {self.selected_feature_names}")

        return X_selected

    def _apply_feature_selection(self, X: np.ndarray) -> np.ndarray:
        """Apply fitted feature selection to new data."""
        if self.variance_selector is None or self.k_best_selector is None:
            raise ValueError("Feature selection not fitted. Call fit() first.")

        X_var = self.variance_selector.transform(X)
        X_selected = self.k_best_selector.transform(X_var)
        return X_selected

    def fit(
        self,
        X: np.ndarray,
        actual_spread: np.ndarray,
        vegas_spread: np.ndarray,
        feature_names: list[str],
        week: Optional[np.ndarray] = None,
        home_rest: Optional[np.ndarray] = None,
        away_rest: Optional[np.ndarray] = None,
        temperature: Optional[np.ndarray] = None,
        wind_speed: Optional[np.ndarray] = None,
        seasons: Optional[list[int]] = None,
    ) -> 'ResidualSpreadModel':
        """
        Fit the residual model.

        Args:
            X: Feature matrix (n_samples, n_features)
            actual_spread: Actual game spread (positive = home team favored)
            vegas_spread: Vegas spread (positive = home team favored)
            feature_names: Names of features in X
            week: Week numbers for situational features
            home_rest: Home team rest days
            away_rest: Away team rest days
            temperature: Game temperature
            wind_speed: Wind speed
            seasons: Training seasons for metadata
        """
        logger.info("Fitting ResidualSpreadModel...")

        # Store original feature names
        self.feature_names = list(feature_names)

        # Compute residuals (target)
        residuals = actual_spread - vegas_spread
        logger.info(f"Residual stats - Mean: {residuals.mean():.3f}, Std: {residuals.std():.3f}")

        # Add situational features
        X_augmented, augmented_names = self._add_situational_features(
            X, self.feature_names,
            week=week,
            vegas_spread=vegas_spread,
            home_rest=home_rest,
            away_rest=away_rest,
            temperature=temperature,
            wind_speed=wind_speed,
        )

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_augmented)

        # Feature selection
        X_selected = self._fit_feature_selection(X_scaled, residuals, augmented_names)

        # Fit model
        self.model = self._create_model()
        self.model.fit(X_selected, residuals)

        # Log coefficients for interpretability
        if hasattr(self.model, 'coef_'):
            coef_importance = sorted(
                zip(self.selected_feature_names, self.model.coef_),
                key=lambda x: abs(x[1]),
                reverse=True
            )
            logger.info("Top feature coefficients:")
            for name, coef in coef_importance[:10]:
                logger.info(f"  {name}: {coef:.4f}")

        # Create metadata (compatible with ModelMetadata dataclass)
        from nfl_bets.models.base import ModelMetrics
        self.metadata = ModelMetadata(
            model_type="residual_spread",
            model_version="1.0.0",
            training_date=datetime.now(),
            training_seasons=seasons or [],
            n_training_samples=len(actual_spread),
            feature_names=self.selected_feature_names,
            hyperparameters={
                'k_features': self.config.k_features,
                'alpha': self.config.alpha,
                'model_type': self.config.model_type,
                'add_situational_features': self.config.add_situational_features,
            },
            metrics=ModelMetrics(
                n_samples=len(actual_spread),
            ),
            calibrated=False,
            data_cutoff_date=datetime.now(),
        )

        self.is_trained = True
        logger.info("ResidualSpreadModel training complete")

        return self

    def predict(
        self,
        X: np.ndarray,
        vegas_spread: np.ndarray,
        week: Optional[np.ndarray] = None,
        home_rest: Optional[np.ndarray] = None,
        away_rest: Optional[np.ndarray] = None,
        temperature: Optional[np.ndarray] = None,
        wind_speed: Optional[np.ndarray] = None,
    ) -> list[ResidualPrediction]:
        """
        Make predictions.

        Args:
            X: Feature matrix
            vegas_spread: Current Vegas spread
            week: Week numbers
            home_rest: Home team rest days
            away_rest: Away team rest days
            temperature: Game temperature
            wind_speed: Wind speed

        Returns:
            List of ResidualPrediction objects
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")

        # Add situational features
        X_augmented, _ = self._add_situational_features(
            X, self.feature_names,
            week=week,
            vegas_spread=vegas_spread,
            home_rest=home_rest,
            away_rest=away_rest,
            temperature=temperature,
            wind_speed=wind_speed,
        )

        # Scale and select features
        X_scaled = self.scaler.transform(X_augmented)
        X_selected = self._apply_feature_selection(X_scaled)

        # Predict residuals
        predicted_residuals = self.model.predict(X_selected)

        # Apply blend factor
        blended_residuals = predicted_residuals * self.config.blend_factor

        # Compute final spreads
        final_spreads = vegas_spread + blended_residuals

        # Create predictions
        predictions = []
        for i in range(len(vegas_spread)):
            # Determine situational edge
            edge = None
            if week is not None:
                if week[i] <= 4:
                    edge = "early_season"
                elif 5 <= week[i] <= 8:
                    edge = "mid_early_season"
                elif week[i] >= 13:
                    edge = "late_season"

            if edge is None and abs(vegas_spread[i]) < 3.5:
                edge = "close_game"
            elif edge is None and abs(vegas_spread[i]) >= 10:
                edge = "large_spread"

            # Confidence based on residual magnitude
            # Larger predicted adjustments = more confident in edge
            confidence = min(0.9, 0.5 + abs(predicted_residuals[i]) / 10)

            predictions.append(ResidualPrediction(
                final_spread=float(final_spreads[i]),
                vegas_spread=float(vegas_spread[i]),
                predicted_residual=float(predicted_residuals[i]),
                confidence=float(confidence),
                situational_edge=edge,
            ))

        return predictions

    def predict_single(
        self,
        features: dict,
        vegas_spread: float,
        week: Optional[int] = None,
        home_rest: Optional[int] = None,
        away_rest: Optional[int] = None,
        temperature: Optional[float] = None,
        wind_speed: Optional[float] = None,
    ) -> ResidualPrediction:
        """Make a single prediction (convenience method)."""

        # Convert features dict to array
        X = np.array([[features.get(name, 0.0) for name in self.feature_names]])

        # Wrap scalars in arrays
        vegas_arr = np.array([vegas_spread])
        week_arr = np.array([week]) if week is not None else None
        home_rest_arr = np.array([home_rest]) if home_rest is not None else None
        away_rest_arr = np.array([away_rest]) if away_rest is not None else None
        temp_arr = np.array([temperature]) if temperature is not None else None
        wind_arr = np.array([wind_speed]) if wind_speed is not None else None

        predictions = self.predict(
            X, vegas_arr,
            week=week_arr,
            home_rest=home_rest_arr,
            away_rest=away_rest_arr,
            temperature=temp_arr,
            wind_speed=wind_arr,
        )

        return predictions[0]

    def evaluate(
        self,
        X: np.ndarray,
        actual_spread: np.ndarray,
        vegas_spread: np.ndarray,
        week: Optional[np.ndarray] = None,
        home_rest: Optional[np.ndarray] = None,
        away_rest: Optional[np.ndarray] = None,
        temperature: Optional[np.ndarray] = None,
        wind_speed: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Evaluate model performance.

        Returns:
            Dictionary with MAE, ATS%, and situational breakdowns
        """
        predictions = self.predict(
            X, vegas_spread,
            week=week,
            home_rest=home_rest,
            away_rest=away_rest,
            temperature=temperature,
            wind_speed=wind_speed,
        )

        final_spreads = np.array([p.final_spread for p in predictions])

        # Core metrics
        model_errors = np.abs(final_spreads - actual_spread)
        vegas_errors = np.abs(vegas_spread - actual_spread)

        model_mae = float(np.mean(model_errors))
        vegas_mae = float(np.mean(vegas_errors))

        # ATS accuracy
        model_ats_correct = np.sum(model_errors < vegas_errors)
        vegas_ats_correct = np.sum(vegas_errors < model_errors)
        ties = np.sum(model_errors == vegas_errors)

        ats_pct = model_ats_correct / (model_ats_correct + vegas_ats_correct) * 100 if (model_ats_correct + vegas_ats_correct) > 0 else 50.0

        results = {
            'model_mae': model_mae,
            'vegas_mae': vegas_mae,
            'mae_improvement': vegas_mae - model_mae,
            'beats_vegas': model_mae < vegas_mae,
            'ats_pct': float(ats_pct),
            'ats_above_breakeven': ats_pct > 52.4,
            'model_wins': int(model_ats_correct),
            'vegas_wins': int(vegas_ats_correct),
            'ties': int(ties),
            'n_samples': len(actual_spread),
        }

        # Situational breakdown
        if week is not None:
            situations = {
                'early_season': (week <= 4),
                'mid_early_season': (week >= 5) & (week <= 8),
                'mid_season': (week >= 9) & (week <= 12),
                'late_season': (week >= 13),
            }

            for name, mask in situations.items():
                if mask.sum() > 0:
                    sit_model_wins = np.sum(model_errors[mask] < vegas_errors[mask])
                    sit_vegas_wins = np.sum(vegas_errors[mask] < model_errors[mask])
                    sit_total = sit_model_wins + sit_vegas_wins
                    if sit_total > 0:
                        results[f'{name}_ats'] = float(sit_model_wins / sit_total * 100)
                        results[f'{name}_n'] = int(mask.sum())

        # Spread magnitude breakdown
        if True:  # Always do this
            spread_mag = np.abs(vegas_spread)
            spreads = {
                'close_games': spread_mag < 3.5,
                'medium_spreads': (spread_mag >= 3.5) & (spread_mag < 7),
                'large_spreads': spread_mag >= 7,
            }

            for name, mask in spreads.items():
                if mask.sum() > 0:
                    sit_model_wins = np.sum(model_errors[mask] < vegas_errors[mask])
                    sit_vegas_wins = np.sum(vegas_errors[mask] < model_errors[mask])
                    sit_total = sit_model_wins + sit_vegas_wins
                    if sit_total > 0:
                        results[f'{name}_ats'] = float(sit_model_wins / sit_total * 100)
                        results[f'{name}_n'] = int(mask.sum())

        return results

    def save(self, path: Path) -> None:
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            'config': self.config,
            'scaler': self.scaler,
            'variance_selector': self.variance_selector,
            'k_best_selector': self.k_best_selector,
            'model': self.model,
            'feature_names': self.feature_names,
            'selected_feature_names': self.selected_feature_names,
            'selected_feature_indices': self.selected_feature_indices,
            'metadata': self.metadata,
            'is_trained': self.is_trained,
        }

        with open(path, 'wb') as f:
            pickle.dump(model_data, f)

        logger.info(f"Saved ResidualSpreadModel to {path}")

    @classmethod
    def load(cls, path: Path) -> 'ResidualSpreadModel':
        """Load model from disk."""
        path = Path(path)

        with open(path, 'rb') as f:
            model_data = pickle.load(f)

        model = cls(config=model_data['config'])
        model.scaler = model_data['scaler']
        model.variance_selector = model_data['variance_selector']
        model.k_best_selector = model_data['k_best_selector']
        model.model = model_data['model']
        model.feature_names = model_data['feature_names']
        model.selected_feature_names = model_data['selected_feature_names']
        model.selected_feature_indices = model_data['selected_feature_indices']
        model.metadata = model_data['metadata']
        model.is_trained = model_data['is_trained']

        logger.info(f"Loaded ResidualSpreadModel from {path}")
        return model
