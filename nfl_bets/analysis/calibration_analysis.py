"""
Calibration analysis for probability predictions.

Provides:
- Reliability diagrams
- Calibration metrics with confidence intervals
- Multi-model calibration comparison
- Trend tracking over time
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from loguru import logger

from nfl_bets.models.evaluation import (
    calculate_calibration_curve,
    expected_calibration_error,
    max_calibration_error,
    brier_score,
    log_loss,
    bootstrap_ci,
)


@dataclass
class ReliabilityDiagram:
    """Data for plotting a reliability diagram."""

    bin_edges: list[float]
    mean_predicted: list[float]
    mean_actual: list[float]
    counts: list[int]
    confidence_intervals: list[tuple[float, float]]  # CI for each bin

    @property
    def n_bins(self) -> int:
        return len(self.mean_predicted)

    @property
    def total_samples(self) -> int:
        return sum(self.counts)

    def get_plotly_traces(self) -> list[dict]:
        """Get Plotly trace data for visualization."""
        # Perfect calibration line
        perfect_line = {
            "x": [0, 1],
            "y": [0, 1],
            "mode": "lines",
            "line": {"dash": "dash", "color": "gray"},
            "name": "Perfect Calibration",
        }

        # Calibration curve
        valid_bins = [i for i, c in enumerate(self.counts) if c > 0]
        calibration = {
            "x": [self.mean_predicted[i] for i in valid_bins],
            "y": [self.mean_actual[i] for i in valid_bins],
            "mode": "lines+markers",
            "marker": {"size": [min(20, max(5, c / 10)) for i, c in enumerate(self.counts) if c > 0]},
            "name": "Model Calibration",
        }

        # Error bars
        error_y = {
            "type": "data",
            "array": [(self.confidence_intervals[i][1] - self.mean_actual[i]) for i in valid_bins],
            "arrayminus": [(self.mean_actual[i] - self.confidence_intervals[i][0]) for i in valid_bins],
            "visible": True,
        }
        calibration["error_y"] = error_y

        return [perfect_line, calibration]

    def to_markdown_table(self) -> str:
        """Generate a markdown table of calibration data."""
        lines = [
            "| Bin | Predicted | Actual | Count | Gap |",
            "|-----|-----------|--------|-------|-----|",
        ]

        for i in range(self.n_bins):
            if self.counts[i] > 0:
                gap = abs(self.mean_predicted[i] - self.mean_actual[i])
                lines.append(
                    f"| {i+1} | {self.mean_predicted[i]:.1%} | "
                    f"{self.mean_actual[i]:.1%} | {self.counts[i]} | {gap:.1%} |"
                )

        return "\n".join(lines)


@dataclass
class CalibrationReport:
    """Complete calibration analysis report."""

    model_name: str
    n_samples: int

    # Core metrics
    ece: float
    ece_ci: tuple[float, float]
    mce: float
    brier_score: float
    brier_ci: tuple[float, float]
    log_loss_value: float

    # Reliability diagram
    reliability_diagram: ReliabilityDiagram

    # Thresholds
    is_well_calibrated: bool  # ECE < 0.05
    is_overconfident: bool  # More extreme predictions than outcomes
    is_underconfident: bool  # Less extreme predictions than outcomes

    # Additional analysis
    sharpness: float  # How spread out are predictions
    resolution: float  # How different are outcomes by bin

    def summary(self) -> str:
        """Generate summary string."""
        status = "✓ WELL CALIBRATED" if self.is_well_calibrated else "✗ NEEDS CALIBRATION"

        lines = [
            f"Calibration Report: {self.model_name}",
            "=" * 50,
            f"Status: {status}",
            "",
            "Core Metrics:",
            f"  ECE: {self.ece:.4f} ({self.ece_ci[0]:.4f} - {self.ece_ci[1]:.4f})",
            f"  MCE: {self.mce:.4f}",
            f"  Brier Score: {self.brier_score:.4f} ({self.brier_ci[0]:.4f} - {self.brier_ci[1]:.4f})",
            f"  Log Loss: {self.log_loss_value:.4f}",
            "",
            "Characterization:",
            f"  Sharpness: {self.sharpness:.4f}",
            f"  Resolution: {self.resolution:.4f}",
        ]

        if self.is_overconfident:
            lines.append("  ⚠ Model is OVERCONFIDENT (predictions too extreme)")
        elif self.is_underconfident:
            lines.append("  ⚠ Model is UNDERCONFIDENT (predictions too conservative)")

        lines.append("")
        lines.append("Reliability Diagram:")
        lines.append(self.reliability_diagram.to_markdown_table())

        return "\n".join(lines)


class CalibrationAnalyzer:
    """
    Comprehensive calibration analysis.

    Analyzes how well predicted probabilities match actual outcomes.
    """

    def __init__(self, n_bins: int = 10, n_bootstrap: int = 1000):
        self.n_bins = n_bins
        self.n_bootstrap = n_bootstrap
        self.logger = logger.bind(component="calibration_analyzer")

    def analyze(
        self,
        predicted_probs: np.ndarray,
        actual_outcomes: np.ndarray,
        model_name: str = "Model",
    ) -> CalibrationReport:
        """
        Perform comprehensive calibration analysis.

        Args:
            predicted_probs: Predicted probabilities (0-1)
            actual_outcomes: Binary actual outcomes (0 or 1)
            model_name: Name for reporting

        Returns:
            CalibrationReport with full analysis
        """
        predicted_probs = np.asarray(predicted_probs).flatten()
        actual_outcomes = np.asarray(actual_outcomes).flatten()

        n_samples = len(predicted_probs)
        self.logger.info(f"Analyzing calibration for {model_name} ({n_samples} samples)")

        # Core metrics
        ece = expected_calibration_error(predicted_probs, actual_outcomes, self.n_bins)
        mce = max_calibration_error(predicted_probs, actual_outcomes, self.n_bins)
        brier = brier_score(predicted_probs, actual_outcomes)
        ll = log_loss(predicted_probs, actual_outcomes)

        # Confidence intervals via bootstrap
        ece_ci = self._bootstrap_ece_ci(predicted_probs, actual_outcomes)
        brier_ci = self._bootstrap_brier_ci(predicted_probs, actual_outcomes)

        # Reliability diagram
        reliability = self._build_reliability_diagram(predicted_probs, actual_outcomes)

        # Characterization
        is_well_calibrated = ece < 0.05
        is_overconfident, is_underconfident = self._check_confidence_pattern(
            predicted_probs, actual_outcomes
        )

        # Sharpness and resolution
        sharpness = self._calculate_sharpness(predicted_probs)
        resolution = self._calculate_resolution(predicted_probs, actual_outcomes)

        return CalibrationReport(
            model_name=model_name,
            n_samples=n_samples,
            ece=ece,
            ece_ci=ece_ci,
            mce=mce,
            brier_score=brier,
            brier_ci=brier_ci,
            log_loss_value=ll,
            reliability_diagram=reliability,
            is_well_calibrated=is_well_calibrated,
            is_overconfident=is_overconfident,
            is_underconfident=is_underconfident,
            sharpness=sharpness,
            resolution=resolution,
        )

    def compare_models(
        self,
        models: dict[str, tuple[np.ndarray, np.ndarray]],
    ) -> dict[str, CalibrationReport]:
        """
        Compare calibration across multiple models.

        Args:
            models: Dict mapping model_name -> (predicted_probs, actual_outcomes)

        Returns:
            Dict of CalibrationReports by model name
        """
        reports = {}
        for name, (probs, outcomes) in models.items():
            reports[name] = self.analyze(probs, outcomes, name)

        # Log comparison
        self.logger.info("Calibration Comparison:")
        for name, report in sorted(reports.items(), key=lambda x: x[1].ece):
            self.logger.info(f"  {name}: ECE={report.ece:.4f}, Brier={report.brier_score:.4f}")

        return reports

    def _build_reliability_diagram(
        self,
        predicted_probs: np.ndarray,
        actual_outcomes: np.ndarray,
    ) -> ReliabilityDiagram:
        """Build reliability diagram data."""
        mean_pred, mean_actual, counts = calculate_calibration_curve(
            predicted_probs, actual_outcomes, self.n_bins
        )

        # Calculate confidence intervals for each bin
        bin_edges = np.linspace(0, 1, self.n_bins + 1)
        bin_indices = np.digitize(predicted_probs, bin_edges[1:-1])

        confidence_intervals = []
        for i in range(self.n_bins):
            mask = bin_indices == i
            if counts[i] > 5:
                bin_outcomes = actual_outcomes[mask]
                _, ci_low, ci_high = bootstrap_ci(bin_outcomes, self.n_bootstrap // 10)
                confidence_intervals.append((ci_low, ci_high))
            else:
                confidence_intervals.append((0.0, 1.0))

        return ReliabilityDiagram(
            bin_edges=bin_edges.tolist(),
            mean_predicted=mean_pred.tolist(),
            mean_actual=mean_actual.tolist(),
            counts=counts.astype(int).tolist(),
            confidence_intervals=confidence_intervals,
        )

    def _bootstrap_ece_ci(
        self,
        predicted_probs: np.ndarray,
        actual_outcomes: np.ndarray,
    ) -> tuple[float, float]:
        """Calculate bootstrap CI for ECE."""
        n = len(predicted_probs)
        ece_values = []

        for _ in range(self.n_bootstrap):
            idx = np.random.randint(0, n, size=n)
            ece = expected_calibration_error(
                predicted_probs[idx], actual_outcomes[idx], self.n_bins
            )
            ece_values.append(ece)

        return (np.percentile(ece_values, 2.5), np.percentile(ece_values, 97.5))

    def _bootstrap_brier_ci(
        self,
        predicted_probs: np.ndarray,
        actual_outcomes: np.ndarray,
    ) -> tuple[float, float]:
        """Calculate bootstrap CI for Brier score."""
        n = len(predicted_probs)
        brier_values = []

        for _ in range(self.n_bootstrap):
            idx = np.random.randint(0, n, size=n)
            bs = brier_score(predicted_probs[idx], actual_outcomes[idx])
            brier_values.append(bs)

        return (np.percentile(brier_values, 2.5), np.percentile(brier_values, 97.5))

    def _check_confidence_pattern(
        self,
        predicted_probs: np.ndarray,
        actual_outcomes: np.ndarray,
    ) -> tuple[bool, bool]:
        """Check if model is over or underconfident."""
        # Split into high confidence (>0.7 or <0.3) and low confidence
        extreme_mask = (predicted_probs > 0.7) | (predicted_probs < 0.3)

        if not np.any(extreme_mask):
            return False, False

        # For extreme predictions, compare predicted vs actual
        extreme_probs = predicted_probs[extreme_mask]
        extreme_outcomes = actual_outcomes[extreme_mask]

        # Convert to "confidence" (distance from 0.5)
        pred_confidence = np.abs(extreme_probs - 0.5)
        actual_confidence = np.abs(extreme_outcomes - 0.5)

        avg_pred_conf = np.mean(pred_confidence)
        avg_actual_conf = np.mean(actual_confidence)

        # Overconfident if predictions more extreme than outcomes
        is_overconfident = avg_pred_conf > avg_actual_conf + 0.05
        is_underconfident = avg_pred_conf < avg_actual_conf - 0.05

        return is_overconfident, is_underconfident

    def _calculate_sharpness(self, predicted_probs: np.ndarray) -> float:
        """
        Calculate sharpness (variance of predictions).

        Higher sharpness = more confident/spread out predictions.
        """
        return float(np.var(predicted_probs))

    def _calculate_resolution(
        self,
        predicted_probs: np.ndarray,
        actual_outcomes: np.ndarray,
    ) -> float:
        """
        Calculate resolution (variance of outcomes by bin).

        Higher resolution = predictions discriminate between outcomes.
        """
        mean_pred, mean_actual, counts = calculate_calibration_curve(
            predicted_probs, actual_outcomes, self.n_bins
        )

        total = np.sum(counts)
        if total == 0:
            return 0.0

        # Base rate
        base_rate = np.mean(actual_outcomes)

        # Weighted variance of bin means from base rate
        weights = counts / total
        resolution = np.sum(weights * (mean_actual - base_rate) ** 2)

        return float(resolution)
