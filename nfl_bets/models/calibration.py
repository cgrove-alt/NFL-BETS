"""
Probability calibration for NFL prediction models.

Provides post-hoc calibration to ensure predicted probabilities
accurately reflect true frequencies. Well-calibrated probabilities
are essential for correct Kelly criterion bet sizing.

Methods:
- Isotonic regression (non-parametric, flexible)
- Platt scaling (logistic regression, parametric)
- Beta calibration (handles well-calibrated models better)
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional, Union

import joblib
import numpy as np
from loguru import logger
from scipy import stats
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


@dataclass
class CalibrationMetrics:
    """Metrics for evaluating calibration quality."""

    # Expected Calibration Error (lower is better)
    ece: float

    # Maximum Calibration Error
    mce: float

    # Brier Score (lower is better)
    brier_score: float

    # Log Loss
    log_loss: float

    # Number of samples
    n_samples: int

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            "ece": self.ece,
            "mce": self.mce,
            "brier_score": self.brier_score,
            "log_loss": self.log_loss,
            "n_samples": self.n_samples,
        }

    @property
    def is_well_calibrated(self) -> bool:
        """Check if ECE is below threshold (0.05)."""
        return self.ece < 0.05


# Default thresholds for honesty layer
ABSTAIN_CONFIDENCE_THRESHOLD = 0.55  # Abstain if confidence < 55%
POORLY_CALIBRATED_ECE_THRESHOLD = 0.15  # ECE > 15% = poorly calibrated


@dataclass
class CalibrationDiagnostics:
    """
    Calibration diagnostics for model transparency (honesty layer).

    Exposes calibration quality metrics and provides abstention guidance.
    """

    # Core metrics
    ece: float  # Expected Calibration Error
    mce: float  # Maximum Calibration Error
    brier_score: float
    log_loss: float

    # Sample info
    n_calibration_samples: int

    # Method info
    calibration_method: str  # "isotonic", "platt", "beta"

    # Quality flags
    is_well_calibrated: bool  # ECE < 0.05

    # Reliability curve data for visualization
    reliability_curve: Optional[dict] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "ece": self.ece,
            "mce": self.mce,
            "brier_score": self.brier_score,
            "log_loss": self.log_loss,
            "n_calibration_samples": self.n_calibration_samples,
            "calibration_method": self.calibration_method,
            "is_well_calibrated": self.is_well_calibrated,
            "reliability_curve": self.reliability_curve,
        }

    @property
    def is_poorly_calibrated(self) -> bool:
        """Check if model is poorly calibrated (ECE > 15%)."""
        return self.ece > POORLY_CALIBRATED_ECE_THRESHOLD

    @classmethod
    def from_metrics(
        cls,
        metrics: CalibrationMetrics,
        method: str,
        reliability_curve: Optional[dict] = None,
    ) -> "CalibrationDiagnostics":
        """Create diagnostics from CalibrationMetrics."""
        return cls(
            ece=metrics.ece,
            mce=metrics.mce,
            brier_score=metrics.brier_score,
            log_loss=metrics.log_loss,
            n_calibration_samples=metrics.n_samples,
            calibration_method=method,
            is_well_calibrated=metrics.is_well_calibrated,
            reliability_curve=reliability_curve,
        )


class ProbabilityCalibrator:
    """
    Post-hoc probability calibration.

    Adjusts predicted probabilities to better match empirical frequencies.
    Essential for accurate bet sizing with Kelly criterion.

    Example:
        >>> calibrator = ProbabilityCalibrator(method="isotonic")
        >>> calibrator.fit(raw_probs, actual_outcomes)
        >>> calibrated_probs = calibrator.calibrate(new_probs)
        >>> metrics = calibrator.evaluate(calibrated_probs, actuals)
    """

    METHODS = ["isotonic", "platt", "beta"]

    def __init__(
        self,
        method: Literal["isotonic", "platt", "beta"] = "isotonic",
    ):
        """
        Initialize the calibrator.

        Args:
            method: Calibration method
                - "isotonic": Non-parametric, flexible (recommended)
                - "platt": Logistic regression, parametric
                - "beta": Beta calibration, handles near-calibrated better
        """
        if method not in self.METHODS:
            raise ValueError(f"Unknown method: {method}. Use one of {self.METHODS}")

        self.method = method
        self.calibrator = None
        self.is_fitted = False
        self.logger = logger.bind(component="calibrator")

    def __getstate__(self):
        """Exclude logger from pickling."""
        state = self.__dict__.copy()
        del state["logger"]
        return state

    def __setstate__(self, state):
        """Restore logger after unpickling."""
        self.__dict__.update(state)
        self.logger = logger.bind(component="calibrator")

    def fit(
        self,
        predicted_probs: np.ndarray,
        actual_outcomes: np.ndarray,
    ) -> "ProbabilityCalibrator":
        """
        Fit the calibrator on validation data.

        Args:
            predicted_probs: Uncalibrated probabilities from model
            actual_outcomes: Binary actual outcomes (0 or 1)

        Returns:
            Self for method chaining
        """
        predicted_probs = np.asarray(predicted_probs).flatten()
        actual_outcomes = np.asarray(actual_outcomes).flatten()

        # Validate inputs
        if len(predicted_probs) != len(actual_outcomes):
            raise ValueError("Predictions and outcomes must have same length")

        if len(predicted_probs) < 10:
            self.logger.warning(
                f"Very few samples ({len(predicted_probs)}) for calibration"
            )

        # Clip probabilities to valid range
        predicted_probs = np.clip(predicted_probs, 0.001, 0.999)

        if self.method == "isotonic":
            self.calibrator = IsotonicRegression(
                out_of_bounds="clip",
                y_min=0.0,
                y_max=1.0,
            )
            self.calibrator.fit(predicted_probs, actual_outcomes)

        elif self.method == "platt":
            # Platt scaling: fit logistic regression on log-odds
            self.calibrator = LogisticRegression(solver="lbfgs", max_iter=1000)
            # Transform to log-odds space
            log_odds = np.log(predicted_probs / (1 - predicted_probs))
            self.calibrator.fit(log_odds.reshape(-1, 1), actual_outcomes)

        elif self.method == "beta":
            # Beta calibration: fits a, b, c in sigmoid(a * log(p/(1-p)) + b)
            self._fit_beta(predicted_probs, actual_outcomes)

        self.is_fitted = True
        self.logger.info(
            f"Calibrator fitted using {self.method} on {len(predicted_probs)} samples"
        )
        return self

    def _fit_beta(
        self,
        predicted_probs: np.ndarray,
        actual_outcomes: np.ndarray,
    ) -> None:
        """Fit beta calibration parameters."""
        from scipy.optimize import minimize

        def beta_loss(params, probs, outcomes):
            a, b, c = params
            log_odds = np.log(probs / (1 - probs))
            calibrated_logits = a * log_odds + b
            calibrated = 1 / (1 + np.exp(-calibrated_logits))
            calibrated = np.clip(calibrated, 1e-7, 1 - 1e-7)
            # Log loss
            loss = -np.mean(
                outcomes * np.log(calibrated)
                + (1 - outcomes) * np.log(1 - calibrated)
            )
            return loss

        # Optimize
        result = minimize(
            beta_loss,
            x0=[1.0, 0.0, 1.0],
            args=(predicted_probs, actual_outcomes),
            method="L-BFGS-B",
            bounds=[(0.1, 10), (-5, 5), (0.1, 10)],
        )
        self.calibrator = {"a": result.x[0], "b": result.x[1], "c": result.x[2]}

    def calibrate(
        self,
        probs: np.ndarray,
    ) -> np.ndarray:
        """
        Apply calibration to probabilities.

        Args:
            probs: Raw probabilities to calibrate

        Returns:
            Calibrated probabilities
        """
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted before calibrating")

        probs = np.asarray(probs).flatten()
        probs = np.clip(probs, 0.001, 0.999)

        if self.method == "isotonic":
            return self.calibrator.predict(probs)

        elif self.method == "platt":
            log_odds = np.log(probs / (1 - probs))
            return self.calibrator.predict_proba(log_odds.reshape(-1, 1))[:, 1]

        elif self.method == "beta":
            a, b = self.calibrator["a"], self.calibrator["b"]
            log_odds = np.log(probs / (1 - probs))
            calibrated_logits = a * log_odds + b
            return 1 / (1 + np.exp(-calibrated_logits))

        return probs

    def evaluate(
        self,
        predicted_probs: np.ndarray,
        actual_outcomes: np.ndarray,
        n_bins: int = 10,
    ) -> CalibrationMetrics:
        """
        Evaluate calibration quality.

        Args:
            predicted_probs: Probabilities to evaluate
            actual_outcomes: Binary actual outcomes
            n_bins: Number of bins for ECE calculation

        Returns:
            CalibrationMetrics with ECE, MCE, Brier score, etc.
        """
        probs = np.asarray(predicted_probs).flatten()
        actuals = np.asarray(actual_outcomes).flatten()

        # ECE and MCE
        ece, mce = self._calculate_ece_mce(probs, actuals, n_bins)

        # Brier score
        brier = float(np.mean((probs - actuals) ** 2))

        # Log loss
        probs_clipped = np.clip(probs, 1e-7, 1 - 1e-7)
        log_loss = float(
            -np.mean(
                actuals * np.log(probs_clipped)
                + (1 - actuals) * np.log(1 - probs_clipped)
            )
        )

        return CalibrationMetrics(
            ece=ece,
            mce=mce,
            brier_score=brier,
            log_loss=log_loss,
            n_samples=len(probs),
        )

    def _calculate_ece_mce(
        self,
        probs: np.ndarray,
        actuals: np.ndarray,
        n_bins: int,
    ) -> tuple[float, float]:
        """Calculate ECE and MCE."""
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(probs, bin_edges[1:-1])

        ece = 0.0
        mce = 0.0
        total = len(probs)

        for i in range(n_bins):
            mask = bin_indices == i
            n_bin = np.sum(mask)
            if n_bin > 0:
                mean_pred = np.mean(probs[mask])
                mean_actual = np.mean(actuals[mask])
                error = np.abs(mean_pred - mean_actual)
                ece += (n_bin / total) * error
                mce = max(mce, error)

        return float(ece), float(mce)

    def should_abstain(
        self,
        calibrated_prob: float,
        confidence_threshold: float = ABSTAIN_CONFIDENCE_THRESHOLD,
        ece_threshold: float = POORLY_CALIBRATED_ECE_THRESHOLD,
        last_ece: Optional[float] = None,
    ) -> bool:
        """
        Determine if model should abstain from making a prediction.

        Part of the "honesty layer" - models should decline predictions when:
        1. Confidence is too close to 50% (coin flip)
        2. Model calibration is poor (high ECE)

        Args:
            calibrated_prob: Calibrated probability (0-1)
            confidence_threshold: Minimum confidence required (default 0.55)
            ece_threshold: Maximum acceptable ECE (default 0.15)
            last_ece: ECE from most recent evaluation (optional)

        Returns:
            True if model should abstain, False if prediction is confident enough
        """
        # Calculate confidence (distance from 0.5, mapped to 0.5-1.0)
        confidence = max(calibrated_prob, 1 - calibrated_prob)

        # Abstain if confidence below threshold
        if confidence < confidence_threshold:
            return True

        # Abstain if model is poorly calibrated (if ECE provided)
        if last_ece is not None and last_ece > ece_threshold:
            return True

        return False

    def get_diagnostics(
        self,
        predicted_probs: np.ndarray,
        actual_outcomes: np.ndarray,
        n_bins: int = 10,
    ) -> CalibrationDiagnostics:
        """
        Get full calibration diagnostics including reliability curve.

        Args:
            predicted_probs: Probabilities to evaluate
            actual_outcomes: Binary actual outcomes
            n_bins: Number of bins for ECE calculation

        Returns:
            CalibrationDiagnostics with all metrics and visualization data
        """
        metrics = self.evaluate(predicted_probs, actual_outcomes, n_bins)
        reliability = reliability_diagram_data(predicted_probs, actual_outcomes, n_bins)

        return CalibrationDiagnostics.from_metrics(
            metrics=metrics,
            method=self.method,
            reliability_curve=reliability,
        )

    def get_calibration_curve(
        self,
        predicted_probs: np.ndarray,
        actual_outcomes: np.ndarray,
        n_bins: int = 10,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get calibration curve data for plotting.

        Args:
            predicted_probs: Predicted probabilities
            actual_outcomes: Binary actual outcomes
            n_bins: Number of bins

        Returns:
            Tuple of (mean_predicted, mean_actual, counts) for each bin
        """
        probs = np.asarray(predicted_probs).flatten()
        actuals = np.asarray(actual_outcomes).flatten()

        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(probs, bin_edges[1:-1])

        mean_predicted = np.zeros(n_bins)
        mean_actual = np.zeros(n_bins)
        counts = np.zeros(n_bins)

        for i in range(n_bins):
            mask = bin_indices == i
            counts[i] = np.sum(mask)
            if counts[i] > 0:
                mean_predicted[i] = np.mean(probs[mask])
                mean_actual[i] = np.mean(actuals[mask])

        return mean_predicted, mean_actual, counts

    def save(self, path: Union[str, Path]) -> None:
        """Save calibrator to disk."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted calibrator")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        save_dict = {
            "method": self.method,
            "calibrator": self.calibrator,
            "is_fitted": self.is_fitted,
        }
        joblib.dump(save_dict, path)
        self.logger.info(f"Calibrator saved to {path}")

    def load(self, path: Union[str, Path]) -> "ProbabilityCalibrator":
        """Load calibrator from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Calibrator file not found: {path}")

        save_dict = joblib.load(path)
        self.method = save_dict["method"]
        self.calibrator = save_dict["calibrator"]
        self.is_fitted = save_dict["is_fitted"]

        self.logger.info(f"Calibrator loaded from {path}")
        return self


class TemperatureScaling:
    """
    Temperature scaling for neural network calibration.

    Simple post-hoc calibration that learns a single temperature
    parameter to scale logits.

    logits_calibrated = logits / temperature
    """

    def __init__(self, initial_temperature: float = 1.0):
        self.temperature = initial_temperature
        self.is_fitted = False
        self.logger = logger.bind(component="temperature_scaling")

    def fit(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
    ) -> "TemperatureScaling":
        """
        Fit optimal temperature on validation data.

        Args:
            logits: Raw logits from model
            labels: Binary labels

        Returns:
            Self for method chaining
        """
        from scipy.optimize import minimize_scalar

        def nll_loss(temperature):
            scaled = logits / temperature
            probs = 1 / (1 + np.exp(-scaled))
            probs = np.clip(probs, 1e-7, 1 - 1e-7)
            loss = -np.mean(
                labels * np.log(probs) + (1 - labels) * np.log(1 - probs)
            )
            return loss

        result = minimize_scalar(nll_loss, bounds=(0.1, 10.0), method="bounded")
        self.temperature = result.x
        self.is_fitted = True

        self.logger.info(f"Temperature scaling fitted: T = {self.temperature:.4f}")
        return self

    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """Apply temperature scaling to logits and return probabilities."""
        if not self.is_fitted:
            raise ValueError("Must fit temperature before calibrating")

        scaled = logits / self.temperature
        return 1 / (1 + np.exp(-scaled))


def calibrate_spread_probabilities(
    predicted_spreads: np.ndarray,
    spread_stds: np.ndarray,
    betting_lines: np.ndarray,
    actual_covers: np.ndarray,
    calibrator: Optional[ProbabilityCalibrator] = None,
) -> tuple[np.ndarray, ProbabilityCalibrator]:
    """
    Calibrate cover probabilities for spread predictions.

    Converts spread predictions + uncertainty to calibrated cover probabilities.

    Args:
        predicted_spreads: Model's predicted spreads
        spread_stds: Uncertainty (std) in spread predictions
        betting_lines: Actual betting lines
        actual_covers: Binary outcomes (1 = home covered)
        calibrator: Optional pre-fitted calibrator

    Returns:
        Tuple of (calibrated_probs, fitted_calibrator)
    """
    # Calculate raw cover probabilities from normal distribution
    # P(cover) = P(actual > line) ≈ P(predicted - std*z > line)
    z_scores = (predicted_spreads - betting_lines) / np.maximum(spread_stds, 0.1)
    raw_probs = stats.norm.cdf(z_scores)

    # Fit calibrator if not provided
    if calibrator is None:
        calibrator = ProbabilityCalibrator(method="isotonic")
        calibrator.fit(raw_probs, actual_covers)

    # Apply calibration
    calibrated_probs = calibrator.calibrate(raw_probs)

    return calibrated_probs, calibrator


def reliability_diagram_data(
    predicted_probs: np.ndarray,
    actual_outcomes: np.ndarray,
    n_bins: int = 10,
) -> dict[str, Any]:
    """
    Generate data for a reliability diagram.

    Args:
        predicted_probs: Predicted probabilities
        actual_outcomes: Binary actual outcomes
        n_bins: Number of bins

    Returns:
        Dictionary with data for plotting
    """
    probs = np.asarray(predicted_probs).flatten()
    actuals = np.asarray(actual_outcomes).flatten()

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_indices = np.digitize(probs, bin_edges[1:-1])

    mean_predicted = []
    mean_actual = []
    counts = []
    errors = []

    for i in range(n_bins):
        mask = bin_indices == i
        n_bin = np.sum(mask)
        counts.append(n_bin)

        if n_bin > 0:
            pred = np.mean(probs[mask])
            actual = np.mean(actuals[mask])
            mean_predicted.append(pred)
            mean_actual.append(actual)
            # Standard error for binomial proportion
            se = np.sqrt(actual * (1 - actual) / n_bin) if n_bin > 1 else 0
            errors.append(1.96 * se)
        else:
            mean_predicted.append(bin_centers[i])
            mean_actual.append(np.nan)
            errors.append(0)

    return {
        "bin_centers": bin_centers.tolist(),
        "mean_predicted": mean_predicted,
        "mean_actual": mean_actual,
        "counts": counts,
        "errors": errors,
        "perfect_line": [0, 1],
    }
