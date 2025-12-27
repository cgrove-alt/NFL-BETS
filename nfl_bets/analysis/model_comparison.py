"""
Model comparison against baselines and Vegas.

Provides:
- Vegas benchmark comparison
- Model vs model comparison
- Statistical significance of differences
- Head-to-head analysis
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from loguru import logger
from scipy import stats


@dataclass
class VegasComparison:
    """Comparison of model performance vs Vegas lines."""

    n_games: int

    # Model metrics
    model_mae: float
    model_ats_wins: int
    model_ats_losses: int
    model_ats_rate: float

    # Vegas metrics (as baseline)
    vegas_mae: float  # How close Vegas lines are to actual results
    vegas_ats_rate: float  # Should be ~50%

    # Comparison
    mae_improvement: float  # Positive = model beats Vegas
    ats_improvement: float
    p_value_ats: float  # Significance vs 50%
    p_value_vs_vegas: float  # Significance vs Vegas rate

    # CLV (Closing Line Value)
    avg_clv: Optional[float] = None
    clv_positive_rate: Optional[float] = None

    @property
    def beats_vegas(self) -> bool:
        """Does model significantly beat Vegas?"""
        return self.p_value_ats < 0.05 and self.model_ats_rate > 0.525

    @property
    def model_roi(self) -> float:
        """Model ROI at -110 juice."""
        if self.model_ats_wins + self.model_ats_losses == 0:
            return 0.0
        profit = self.model_ats_wins * (100/110) - self.model_ats_losses
        staked = self.model_ats_wins + self.model_ats_losses
        return profit / staked

    def summary(self) -> str:
        """Generate summary string."""
        status = "✓ BEATS VEGAS" if self.beats_vegas else "✗ Does not beat Vegas"

        lines = [
            "Model vs Vegas Comparison",
            "=" * 50,
            f"Status: {status}",
            f"Games Analyzed: {self.n_games:,}",
            "",
            "Performance Metrics:",
            f"  Model MAE: {self.model_mae:.3f}",
            f"  Vegas MAE: {self.vegas_mae:.3f}",
            f"  MAE Improvement: {self.mae_improvement:+.3f} ({-100*self.mae_improvement/self.vegas_mae:+.1f}%)",
            "",
            "Against The Spread:",
            f"  Model: {self.model_ats_wins}-{self.model_ats_losses} ({self.model_ats_rate:.1%})",
            f"  Model ROI: {self.model_roi:+.1%}",
            f"  p-value vs 50%: {self.p_value_ats:.4f}",
            f"  p-value vs Vegas: {self.p_value_vs_vegas:.4f}",
        ]

        if self.avg_clv is not None:
            lines.append("")
            lines.append("Closing Line Value:")
            lines.append(f"  Average CLV: {self.avg_clv:+.2f} points")
            lines.append(f"  CLV+ Rate: {self.clv_positive_rate:.1%}")

        return "\n".join(lines)


@dataclass
class ModelVsModelComparison:
    """Head-to-head comparison of two models."""

    model_a_name: str
    model_b_name: str
    n_samples: int

    # Model A metrics
    model_a_mae: float
    model_a_ats_rate: float
    model_a_roi: float

    # Model B metrics
    model_b_mae: float
    model_b_ats_rate: float
    model_b_roi: float

    # Comparison
    mae_diff: float  # A - B, negative = A better
    ats_diff: float  # A - B, positive = A better
    roi_diff: float

    # Statistics
    p_value_mae: float
    p_value_ats: float
    correlation: float  # How correlated are predictions

    @property
    def winner(self) -> str:
        """Which model is better overall."""
        a_score = 0
        b_score = 0

        if self.mae_diff < 0:
            a_score += 1
        else:
            b_score += 1

        if self.ats_diff > 0:
            a_score += 1
        else:
            b_score += 1

        if self.roi_diff > 0:
            a_score += 1
        else:
            b_score += 1

        if a_score > b_score:
            return self.model_a_name
        elif b_score > a_score:
            return self.model_b_name
        else:
            return "TIE"

    def summary(self) -> str:
        """Generate comparison summary."""
        lines = [
            f"Model Comparison: {self.model_a_name} vs {self.model_b_name}",
            "=" * 60,
            f"Samples: {self.n_samples:,}",
            f"Winner: {self.winner}",
            "",
            "| Metric | " + self.model_a_name + " | " + self.model_b_name + " | Diff | p-value |",
            "|--------|" + "-" * (len(self.model_a_name) + 2) + "|" + "-" * (len(self.model_b_name) + 2) + "|------|---------|",
            f"| MAE | {self.model_a_mae:.3f} | {self.model_b_mae:.3f} | {self.mae_diff:+.3f} | {self.p_value_mae:.3f} |",
            f"| ATS% | {self.model_a_ats_rate:.1%} | {self.model_b_ats_rate:.1%} | {self.ats_diff:+.1%} | {self.p_value_ats:.3f} |",
            f"| ROI | {self.model_a_roi:+.1%} | {self.model_b_roi:+.1%} | {self.roi_diff:+.1%} | - |",
            "",
            f"Prediction Correlation: {self.correlation:.3f}",
        ]

        return "\n".join(lines)


@dataclass
class ComparisonResult:
    """Complete model comparison results."""

    model_name: str
    vegas_comparison: Optional[VegasComparison] = None
    baseline_comparisons: dict = field(default_factory=dict)

    def summary(self) -> str:
        """Generate complete summary."""
        lines = [
            f"Complete Comparison Analysis: {self.model_name}",
            "=" * 70,
            "",
        ]

        if self.vegas_comparison:
            lines.append(self.vegas_comparison.summary())
            lines.append("")

        for name, comparison in self.baseline_comparisons.items():
            lines.append("-" * 50)
            lines.append(comparison.summary())
            lines.append("")

        return "\n".join(lines)


class VegasBenchmark:
    """
    Benchmark model against Vegas lines.

    Vegas is the gold standard for prediction accuracy.
    A profitable model must beat Vegas-implied probabilities.
    """

    def __init__(self, juice: float = 0.0454):
        self.juice = juice
        self.breakeven_rate = 0.5 + juice / 2
        self.logger = logger.bind(component="vegas_benchmark")

    def compare(
        self,
        predictions: np.ndarray,
        vegas_lines: np.ndarray,
        actuals: np.ndarray,
        opening_lines: Optional[np.ndarray] = None,
    ) -> VegasComparison:
        """
        Compare model predictions against Vegas.

        Args:
            predictions: Model's predicted spreads/totals
            vegas_lines: Vegas closing lines
            actuals: Actual game results
            opening_lines: Optional opening lines for CLV

        Returns:
            VegasComparison with full analysis
        """
        predictions = np.asarray(predictions)
        vegas_lines = np.asarray(vegas_lines)
        actuals = np.asarray(actuals)

        n_games = len(predictions)
        self.logger.info(f"Comparing {n_games} games against Vegas")

        # MAE calculations
        model_mae = float(np.mean(np.abs(predictions - actuals)))
        vegas_mae = float(np.mean(np.abs(vegas_lines - actuals)))
        mae_improvement = vegas_mae - model_mae  # Positive = model better

        # ATS calculations
        # Model picks: if model predicts lower margin than line, pick home
        model_picks_home = predictions < vegas_lines

        # Actual covers: home team covers if actual margin > line
        home_covers = actuals > vegas_lines

        # Model ATS
        model_correct = model_picks_home == home_covers
        model_ats_wins = int(np.sum(model_correct))
        model_ats_losses = n_games - model_ats_wins
        model_ats_rate = model_ats_wins / n_games

        # Vegas ATS (baseline ~50%)
        vegas_ats_rate = 0.5  # By definition

        # Statistical tests
        result_vs_50 = stats.binomtest(model_ats_wins, n_games, 0.5, alternative='greater')
        p_value_ats = result_vs_50.pvalue

        # McNemar's test would be better here but we don't have paired data
        p_value_vs_vegas = p_value_ats  # Simplified

        # CLV calculation
        avg_clv = None
        clv_positive_rate = None
        if opening_lines is not None:
            opening_lines = np.asarray(opening_lines)
            # CLV = how much better is our line vs closing
            # For home bets: lower line is better
            # For away bets: higher line is better
            clv = np.where(
                model_picks_home,
                vegas_lines - opening_lines,  # Got worse line = negative
                opening_lines - vegas_lines,
            )
            # This is simplified; true CLV compares line we bet at vs closing
            avg_clv = float(np.mean(clv))
            clv_positive_rate = float(np.mean(clv > 0))

        return VegasComparison(
            n_games=n_games,
            model_mae=model_mae,
            model_ats_wins=model_ats_wins,
            model_ats_losses=model_ats_losses,
            model_ats_rate=model_ats_rate,
            vegas_mae=vegas_mae,
            vegas_ats_rate=vegas_ats_rate,
            mae_improvement=mae_improvement,
            ats_improvement=model_ats_rate - vegas_ats_rate,
            p_value_ats=p_value_ats,
            p_value_vs_vegas=p_value_vs_vegas,
            avg_clv=avg_clv,
            clv_positive_rate=clv_positive_rate,
        )


class ModelComparison:
    """
    Compare models against each other and baselines.
    """

    def __init__(self):
        self.logger = logger.bind(component="model_comparison")

    def compare_models(
        self,
        model_a_predictions: np.ndarray,
        model_b_predictions: np.ndarray,
        actuals: np.ndarray,
        lines: np.ndarray,
        model_a_name: str = "Model A",
        model_b_name: str = "Model B",
    ) -> ModelVsModelComparison:
        """
        Compare two models head-to-head.

        Args:
            model_a_predictions: Predictions from model A
            model_b_predictions: Predictions from model B
            actuals: Actual outcomes
            lines: Betting lines
            model_a_name: Name of model A
            model_b_name: Name of model B

        Returns:
            ModelVsModelComparison with analysis
        """
        model_a_predictions = np.asarray(model_a_predictions)
        model_b_predictions = np.asarray(model_b_predictions)
        actuals = np.asarray(actuals)
        lines = np.asarray(lines)

        n_samples = len(actuals)

        # MAE
        mae_a = float(np.mean(np.abs(model_a_predictions - actuals)))
        mae_b = float(np.mean(np.abs(model_b_predictions - actuals)))

        # ATS
        a_picks_home = model_a_predictions < lines
        b_picks_home = model_b_predictions < lines
        home_covers = actuals > lines

        a_correct = a_picks_home == home_covers
        b_correct = b_picks_home == home_covers

        ats_rate_a = float(np.mean(a_correct))
        ats_rate_b = float(np.mean(b_correct))

        # ROI
        def calc_roi(correct_mask):
            wins = np.sum(correct_mask)
            losses = len(correct_mask) - wins
            profit = wins * (100/110) - losses
            return profit / len(correct_mask) if len(correct_mask) > 0 else 0.0

        roi_a = calc_roi(a_correct)
        roi_b = calc_roi(b_correct)

        # Statistical tests
        # Paired t-test for MAE
        errors_a = np.abs(model_a_predictions - actuals)
        errors_b = np.abs(model_b_predictions - actuals)
        _, p_value_mae = stats.ttest_rel(errors_a, errors_b)

        # McNemar's test for ATS
        # Contingency: A right B wrong, A wrong B right
        a_right_b_wrong = np.sum(a_correct & ~b_correct)
        a_wrong_b_right = np.sum(~a_correct & b_correct)

        if a_right_b_wrong + a_wrong_b_right > 0:
            mcnemar_stat = (abs(a_right_b_wrong - a_wrong_b_right) - 1) ** 2 / (a_right_b_wrong + a_wrong_b_right)
            p_value_ats = 1 - stats.chi2.cdf(mcnemar_stat, 1)
        else:
            p_value_ats = 1.0

        # Correlation
        correlation = float(np.corrcoef(model_a_predictions, model_b_predictions)[0, 1])

        return ModelVsModelComparison(
            model_a_name=model_a_name,
            model_b_name=model_b_name,
            n_samples=n_samples,
            model_a_mae=mae_a,
            model_a_ats_rate=ats_rate_a,
            model_a_roi=roi_a,
            model_b_mae=mae_b,
            model_b_ats_rate=ats_rate_b,
            model_b_roi=roi_b,
            mae_diff=mae_a - mae_b,
            ats_diff=ats_rate_a - ats_rate_b,
            roi_diff=roi_a - roi_b,
            p_value_mae=p_value_mae,
            p_value_ats=p_value_ats,
            correlation=correlation,
        )

    def compare_to_baselines(
        self,
        model_predictions: np.ndarray,
        actuals: np.ndarray,
        lines: np.ndarray,
        model_name: str = "Model",
    ) -> ComparisonResult:
        """
        Compare model to standard baselines.

        Baselines:
        - Random (50% picks)
        - Always home
        - Always away
        - Vegas line (as prediction)

        Args:
            model_predictions: Model predictions
            actuals: Actual outcomes
            lines: Betting lines
            model_name: Model name for reporting

        Returns:
            ComparisonResult with all comparisons
        """
        model_predictions = np.asarray(model_predictions)
        actuals = np.asarray(actuals)
        lines = np.asarray(lines)

        n = len(actuals)

        # Vegas comparison
        vegas_benchmark = VegasBenchmark()
        vegas_comparison = vegas_benchmark.compare(model_predictions, lines, actuals)

        # Baseline comparisons
        baseline_comparisons = {}

        # Vegas as predictor
        baseline_comparisons["vs Vegas (as predictor)"] = self.compare_models(
            model_predictions,
            lines,  # Vegas line as prediction
            actuals,
            lines,
            model_name,
            "Vegas",
        )

        # Random baseline (simulate)
        np.random.seed(42)
        random_predictions = lines + np.random.normal(0, 3, n)
        baseline_comparisons["vs Random"] = self.compare_models(
            model_predictions,
            random_predictions,
            actuals,
            lines,
            model_name,
            "Random",
        )

        # Home bias baseline (always predict home wins by 3)
        home_bias_predictions = lines - 3
        baseline_comparisons["vs Home Bias"] = self.compare_models(
            model_predictions,
            home_bias_predictions,
            actuals,
            lines,
            model_name,
            "Home Bias",
        )

        return ComparisonResult(
            model_name=model_name,
            vegas_comparison=vegas_comparison,
            baseline_comparisons=baseline_comparisons,
        )
