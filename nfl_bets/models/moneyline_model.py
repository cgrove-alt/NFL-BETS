"""
XGBoost classifier model for NFL moneyline predictions.

Predicts the probability of each team winning outright:
- Binary classification: home win (1) or away win (0)
- Uses same features as spread model
- Outputs calibrated win probabilities

The model provides:
- Win probability for each team
- Confidence estimates
- Value detection for moneyline bets
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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, TimeSeriesSplit, StratifiedKFold
from sklearn.metrics import log_loss, accuracy_score

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
class MoneylinePrediction:
    """Container for a single moneyline prediction."""

    game_id: str
    home_team: str
    away_team: str

    # Win probabilities
    home_win_prob: float
    away_win_prob: float

    # Confidence
    confidence: float  # Max of win probs (0.5-1.0)

    # Betting context
    home_odds: Optional[int] = None  # American odds
    away_odds: Optional[int] = None
    home_implied_prob: Optional[float] = None
    away_implied_prob: Optional[float] = None
    home_edge: Optional[float] = None
    away_edge: Optional[float] = None

    # Recommendation
    pick: Optional[str] = None  # "HOME" or "AWAY"
    edge: Optional[float] = None  # Edge on the pick

    # Metadata
    model_version: str = ""
    prediction_time: datetime = None

    def __post_init__(self):
        if self.prediction_time is None:
            self.prediction_time = datetime.now()
        # Set pick based on highest edge
        if self.home_edge is not None and self.away_edge is not None:
            if self.home_edge > self.away_edge and self.home_edge > 0:
                self.pick = "HOME"
                self.edge = self.home_edge
            elif self.away_edge > self.home_edge and self.away_edge > 0:
                self.pick = "AWAY"
                self.edge = self.away_edge

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "game_id": self.game_id,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "home_win_prob": self.home_win_prob,
            "away_win_prob": self.away_win_prob,
            "confidence": self.confidence,
            "home_odds": self.home_odds,
            "away_odds": self.away_odds,
            "home_implied_prob": self.home_implied_prob,
            "away_implied_prob": self.away_implied_prob,
            "home_edge": self.home_edge,
            "away_edge": self.away_edge,
            "pick": self.pick,
            "edge": self.edge,
            "model_version": self.model_version,
            "prediction_time": self.prediction_time.isoformat(),
        }


class MoneylineModel(BaseModel):
    """
    Binary classification model for NFL moneyline prediction.

    Uses weighted combination of XGBoost, LightGBM, and Logistic Regression
    for robust win probability predictions with calibration.

    Example:
        >>> model = MoneylineModel()
        >>> model.train(X_train, y_train)
        >>> prediction = model.predict_game(
        ...     features, game_id="2024_01_KC_BAL",
        ...     home_team="BAL", away_team="KC",
        ...     home_odds=-150, away_odds=+130
        ... )
        >>> print(f"Home win prob: {prediction.home_win_prob:.1%}")
    """

    MODEL_TYPE = "moneyline_classifier"
    VERSION = "1.0.0"

    # Default ensemble weights
    DEFAULT_WEIGHTS = {
        "xgb": 0.50,
        "lgb": 0.35,
        "lr": 0.15,  # Logistic regression
    }

    # Default XGBoost hyperparameters
    XGB_PARAMS = {
        "n_estimators": 500,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
    }

    # Default LightGBM hyperparameters
    LGB_PARAMS = {
        "n_estimators": 500,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 20,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
        "objective": "binary",
    }

    # Default Logistic Regression hyperparameters
    LR_PARAMS = {
        "C": 1.0,
        "max_iter": 1000,
        "random_state": 42,
    }

    def __init__(
        self,
        ensemble_weights: Optional[dict[str, float]] = None,
        xgb_params: Optional[dict] = None,
        lgb_params: Optional[dict] = None,
        lr_params: Optional[dict] = None,
        use_calibration: bool = True,
    ):
        """
        Initialize the moneyline model.

        Args:
            ensemble_weights: Weights for each model in ensemble
            xgb_params: XGBoost hyperparameters
            lgb_params: LightGBM hyperparameters
            lr_params: Logistic regression hyperparameters
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
        self.lr_model = None

        # Store hyperparameters
        self.xgb_params = {**self.XGB_PARAMS, **(xgb_params or {})}
        self.lgb_params = {**self.LGB_PARAMS, **(lgb_params or {})}
        self.lr_params = {**self.LR_PARAMS, **(lr_params or {})}

        # Calibrator for win probabilities
        self.calibrator: Optional[ProbabilityCalibrator] = None

        self.logger = logger.bind(model="moneyline_classifier")

    def train(
        self,
        X: Union[pl.DataFrame, np.ndarray],
        y: Union[pl.Series, np.ndarray],
        validation_data: Optional[tuple] = None,
        early_stopping_rounds: int = 50,
        fit_calibrator: bool = True,
    ) -> "MoneylineModel":
        """
        Train the ensemble model.

        Args:
            X: Feature matrix
            y: Target values (1 = home win, 0 = away win)
            validation_data: Optional (X_val, y_val) for early stopping
            early_stopping_rounds: Rounds for early stopping
            fit_calibrator: Whether to fit probability calibrator

        Returns:
            Self for method chaining
        """
        X_arr = self._prepare_features(X)
        y_arr = y.to_numpy() if isinstance(y, pl.Series) else np.asarray(y)

        self.logger.info(f"Training moneyline model on {len(y_arr)} samples")

        # Split for validation if not provided
        if validation_data is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X_arr, y_arr, test_size=0.2, random_state=42, stratify=y_arr
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

        # Train Logistic Regression
        if self.weights.get("lr", 0) > 0:
            self._train_lr(X_train, y_train)

        # Re-normalize weights after potential model exclusions
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}

        # Fit calibrator if requested
        if fit_calibrator and self.use_calibration:
            self._fit_calibrator(X_val, y_val)

        self.is_fitted = True

        # Calculate validation metrics
        val_probs = self._ensemble_predict_proba(X_val)
        val_preds = (val_probs >= 0.5).astype(int)
        accuracy = np.mean(val_preds == y_val)
        self.logger.info(f"Validation accuracy: {accuracy:.1%}")

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
                "lr_params": self.lr_params,
            },
            metrics=ModelMetrics(
                mae=0.0,  # Not applicable for classification
                rmse=0.0,
                r2=0.0,
                accuracy=float(np.mean(val_preds == y_val)),
                ats_wins=int(np.sum(val_preds == y_val)),
                ats_losses=len(y_val) - int(np.sum(val_preds == y_val)),
                ats_pushes=0,
                n_samples=len(y_arr),
            ),
            calibrated=self.calibrator is not None,
        )

        self.logger.info("Moneyline model training complete")
        return self

    def _train_xgb(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        early_stopping_rounds: int,
    ) -> None:
        """Train XGBoost classifier."""
        self.logger.debug("Training XGBoost classifier...")

        params = self.xgb_params.copy()
        params["early_stopping_rounds"] = early_stopping_rounds

        self.xgb_model = xgb.XGBClassifier(**params)
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
        """Train LightGBM classifier."""
        self.logger.debug("Training LightGBM classifier...")

        self.lgb_model = lgb.LGBMClassifier(**self.lgb_params)

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

    def _train_lr(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> None:
        """Train Logistic Regression model."""
        self.logger.debug("Training Logistic Regression...")
        from sklearn.preprocessing import StandardScaler

        # Scale features for LR
        self._lr_scaler = StandardScaler()
        X_scaled = self._lr_scaler.fit_transform(X_train)

        self.lr_model = LogisticRegression(**self.lr_params)
        self.lr_model.fit(X_scaled, y_train)

    def _fit_calibrator(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> None:
        """Fit probability calibrator on validation data."""
        self.logger.debug("Fitting probability calibrator...")

        # Generate raw probability predictions
        raw_probs = self._ensemble_predict_proba(X_val)

        self.calibrator = ProbabilityCalibrator(method="isotonic")
        self.calibrator.fit(raw_probs, y_val)

    def _ensemble_predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble probability predictions (home win prob)."""
        proba = np.zeros(len(X))

        if self.xgb_model is not None and self.weights.get("xgb", 0) > 0:
            proba += self.weights["xgb"] * self.xgb_model.predict_proba(X)[:, 1]

        if self.lgb_model is not None and self.weights.get("lgb", 0) > 0:
            proba += self.weights["lgb"] * self.lgb_model.predict_proba(X)[:, 1]

        if self.lr_model is not None and self.weights.get("lr", 0) > 0:
            X_scaled = self._lr_scaler.transform(X)
            proba += self.weights["lr"] * self.lr_model.predict_proba(X_scaled)[:, 1]

        return proba

    def predict(
        self,
        X: Union[pl.DataFrame, np.ndarray],
    ) -> np.ndarray:
        """
        Predict game winners.

        Args:
            X: Feature matrix

        Returns:
            Array of predictions (1 = home win, 0 = away win)
        """
        self._validate_fitted()
        X_arr = self._prepare_features(X)
        probs = self.predict_proba(X_arr)
        return (probs >= 0.5).astype(int)

    def predict_proba(
        self,
        X: Union[pl.DataFrame, np.ndarray],
    ) -> np.ndarray:
        """
        Predict home team win probabilities.

        Args:
            X: Feature matrix

        Returns:
            Array of home win probabilities
        """
        self._validate_fitted()
        X_arr = self._prepare_features(X)

        raw_probs = self._ensemble_predict_proba(X_arr)

        # Apply calibration if available
        if self.calibrator is not None:
            return self.calibrator.calibrate(raw_probs)

        return raw_probs

    def predict_game(
        self,
        features: Union[dict, pl.DataFrame, np.ndarray],
        game_id: str,
        home_team: str,
        away_team: str,
        home_odds: Optional[int] = None,
        away_odds: Optional[int] = None,
    ) -> MoneylinePrediction:
        """
        Make a single game prediction with full context.

        Args:
            features: Feature values for the game
            game_id: Unique game identifier
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            home_odds: Home team American odds (e.g., -150)
            away_odds: Away team American odds (e.g., +130)

        Returns:
            MoneylinePrediction with all prediction details
        """
        self._validate_fitted()

        # Prepare features
        if isinstance(features, dict):
            X = np.array([[features.get(f, 0.0) for f in self.feature_names]])
        elif isinstance(features, pl.DataFrame):
            X = features.select(self.feature_names).to_numpy()
        else:
            X = np.atleast_2d(features)

        # Get win probability
        home_win_prob = float(self.predict_proba(X)[0])
        away_win_prob = 1.0 - home_win_prob

        # Calculate implied probabilities and edges if odds provided
        home_implied_prob = None
        away_implied_prob = None
        home_edge = None
        away_edge = None

        if home_odds is not None and away_odds is not None:
            home_implied_prob = self._american_to_prob(home_odds)
            away_implied_prob = self._american_to_prob(away_odds)

            # Remove vig for fair comparison
            total_implied = home_implied_prob + away_implied_prob
            home_no_vig = home_implied_prob / total_implied
            away_no_vig = away_implied_prob / total_implied

            home_edge = home_win_prob - home_no_vig
            away_edge = away_win_prob - away_no_vig

        return MoneylinePrediction(
            game_id=game_id,
            home_team=home_team,
            away_team=away_team,
            home_win_prob=home_win_prob,
            away_win_prob=away_win_prob,
            confidence=max(home_win_prob, away_win_prob),
            home_odds=home_odds,
            away_odds=away_odds,
            home_implied_prob=home_implied_prob,
            away_implied_prob=away_implied_prob,
            home_edge=home_edge,
            away_edge=away_edge,
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

        # LR importance (use absolute coefficients)
        if self.lr_model is not None and self.weights.get("lr", 0) > 0:
            lr_imp = np.abs(self.lr_model.coef_[0])
            lr_imp = lr_imp / np.sum(lr_imp)  # Normalize
            for i, name in enumerate(self.feature_names):
                importance[name] = importance.get(name, 0) + self.weights["lr"] * lr_imp[i]

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
        study_name: str = "moneyline_model_optimization",
    ) -> tuple["MoneylineModel", dict[str, Any]]:
        """
        Optimize hyperparameters using Optuna with stratified time-series CV.

        Uses TPE sampler for efficient search, optimizing log loss.

        Args:
            X: Feature matrix
            y: Target values (1=home win, 0=away win)
            n_trials: Number of optimization trials
            n_cv_splits: Number of CV splits
            timeout: Optional timeout in seconds
            study_name: Name for the Optuna study

        Returns:
            Tuple of (optimized model, best hyperparameters)
        """
        if not HAS_OPTUNA:
            raise ImportError("Optuna is required. pip install optuna")

        # Prepare data
        if isinstance(X, pl.DataFrame):
            X_arr = X.to_numpy()
            feature_names = X.columns
        else:
            X_arr = np.asarray(X)
            feature_names = None

        y_arr = y.to_numpy() if isinstance(y, pl.Series) else np.asarray(y)

        logger.info(f"Starting Optuna optimization for moneyline with {n_trials} trials")

        def objective(trial: optuna.Trial) -> float:
            """Objective function for Optuna optimization."""
            # XGBoost hyperparameters
            xgb_params = {
                "n_estimators": trial.suggest_int("xgb_n_estimators", 100, 800),
                "max_depth": trial.suggest_int("xgb_max_depth", 3, 8),
                "learning_rate": trial.suggest_float("xgb_learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("xgb_subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("xgb_colsample_bytree", 0.5, 1.0),
                "min_child_weight": trial.suggest_int("xgb_min_child_weight", 1, 10),
                "reg_alpha": trial.suggest_float("xgb_reg_alpha", 1e-3, 10.0, log=True),
                "reg_lambda": trial.suggest_float("xgb_reg_lambda", 1e-3, 10.0, log=True),
                "random_state": 42,
                "n_jobs": -1,
                "objective": "binary:logistic",
                "eval_metric": "logloss",
            }

            # LightGBM hyperparameters
            lgb_params = {
                "n_estimators": trial.suggest_int("lgb_n_estimators", 100, 800),
                "max_depth": trial.suggest_int("lgb_max_depth", 3, 8),
                "learning_rate": trial.suggest_float("lgb_learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("lgb_subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("lgb_colsample_bytree", 0.5, 1.0),
                "min_child_samples": trial.suggest_int("lgb_min_child_samples", 5, 50),
                "reg_alpha": trial.suggest_float("lgb_reg_alpha", 1e-3, 10.0, log=True),
                "reg_lambda": trial.suggest_float("lgb_reg_lambda", 1e-3, 10.0, log=True),
                "random_state": 42,
                "n_jobs": -1,
                "verbose": -1,
                "objective": "binary",
            }

            # LR hyperparameters
            lr_params = {
                "C": trial.suggest_float("lr_C", 0.01, 100.0, log=True),
                "max_iter": 1000,
                "random_state": 42,
            }

            # Ensemble weights
            xgb_weight = trial.suggest_float("xgb_weight", 0.2, 0.7)
            lgb_weight = trial.suggest_float("lgb_weight", 0.1, 0.5)
            lr_weight = 1.0 - xgb_weight - lgb_weight
            if lr_weight < 0.05:
                return float("inf")

            weights = {
                "xgb": xgb_weight,
                "lgb": lgb_weight,
                "lr": lr_weight,
            }

            # Stratified K-Fold for classification
            skf = StratifiedKFold(n_splits=n_cv_splits, shuffle=True, random_state=42)
            cv_scores = []

            for train_idx, val_idx in skf.split(X_arr, y_arr):
                X_train, X_val = X_arr[train_idx], X_arr[val_idx]
                y_train, y_val = y_arr[train_idx], y_arr[val_idx]

                model = cls(
                    ensemble_weights=weights,
                    xgb_params=xgb_params,
                    lgb_params=lgb_params,
                    lr_params=lr_params,
                    use_calibration=False,
                )

                try:
                    model.train(
                        X_train, y_train,
                        validation_data=(X_val, y_val),
                        early_stopping_rounds=30,
                        fit_calibrator=False,
                    )

                    # Calculate log loss on validation
                    probs = model.predict_proba(X_val)
                    logloss = log_loss(y_val, probs)
                    cv_scores.append(logloss)

                except Exception as e:
                    logger.warning(f"Trial failed: {e}")
                    return float("inf")

            mean_logloss = np.mean(cv_scores)
            trial.set_user_attr("cv_scores", cv_scores)

            return mean_logloss

        # Create study
        sampler = TPESampler(seed=42)
        study = optuna.create_study(
            study_name=study_name,
            direction="minimize",
            sampler=sampler,
        )

        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True,
        )

        best_params = study.best_params
        best_logloss = study.best_value

        logger.info(f"Best log loss: {best_logloss:.4f}")
        logger.info(f"Best params: {best_params}")

        # Build final model
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
            "objective": "binary:logistic",
            "eval_metric": "logloss",
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
            "objective": "binary",
        }

        lr_params = {
            "C": best_params["lr_C"],
            "max_iter": 1000,
            "random_state": 42,
        }

        weights = {
            "xgb": best_params["xgb_weight"],
            "lgb": best_params["lgb_weight"],
            "lr": 1.0 - best_params["xgb_weight"] - best_params["lgb_weight"],
        }

        final_model = cls(
            ensemble_weights=weights,
            xgb_params=xgb_params,
            lgb_params=lgb_params,
            lr_params=lr_params,
            use_calibration=True,
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_arr, y_arr, test_size=0.2, random_state=42, stratify=y_arr
        )

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
            "best_logloss": best_logloss,
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
            "lr_model": self.lr_model,
            "lr_scaler": getattr(self, "_lr_scaler", None),
            "calibrator": self.calibrator,
            "feature_names": self.feature_names,
            "metadata": self.metadata,
            "is_fitted": self.is_fitted,
            "xgb_params": self.xgb_params,
            "lgb_params": self.lgb_params,
            "lr_params": self.lr_params,
        }

        joblib.dump(save_dict, path)
        self.logger.info(f"Model saved to {path}")

    def load(self, path: Union[str, Path]) -> "MoneylineModel":
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
        self.lr_model = save_dict["lr_model"]
        self._lr_scaler = save_dict.get("lr_scaler")
        self.calibrator = save_dict["calibrator"]
        self.feature_names = save_dict["feature_names"]
        self.metadata = save_dict.get("metadata")
        self.is_fitted = save_dict.get("is_fitted", True)
        self.xgb_params = save_dict.get("xgb_params", self.XGB_PARAMS)
        self.lgb_params = save_dict.get("lgb_params", self.LGB_PARAMS)
        self.lr_params = save_dict.get("lr_params", self.LR_PARAMS)

        self.logger.info(f"Model loaded from {path}")
        return self
