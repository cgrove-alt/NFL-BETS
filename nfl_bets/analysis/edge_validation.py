"""
Edge validation and threshold optimization.

Provides:
- Optimal edge threshold discovery
- Edge bucket win rate analysis
- Statistical significance of edges
- CLV (Closing Line Value) validation
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from loguru import logger
from scipy import stats


@dataclass
class EdgeBucket:
    """Results for a specific edge range."""

    edge_min: float
    edge_max: float
    n_bets: int
    wins: int
    losses: int
    total_profit: float

    # Statistical measures
    win_rate: float
    roi: float
    p_value: float  # Significance vs 52.5%
    ci_lower: float
    ci_upper: float

    @property
    def is_significant(self) -> bool:
        """Is this edge bucket statistically significant?"""
        return self.p_value < 0.05 and self.ci_lower > 0.525


@dataclass
class EdgeBucketAnalysis:
    """Complete edge bucket breakdown."""

    buckets: list[EdgeBucket]
    optimal_min_edge: float
    optimal_roi: float

    # Summary stats
    total_bets: int
    overall_win_rate: float
    overall_roi: float

    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            "Edge Bucket Analysis",
            "=" * 60,
            f"Optimal Min Edge: {self.optimal_min_edge:.1%} (ROI: {self.optimal_roi:+.1%})",
            "",
            "Breakdown by Edge:",
            "| Edge Range | Bets | Win Rate | ROI | p-value | Significant |",
            "|------------|------|----------|-----|---------|-------------|",
        ]

        for b in self.buckets:
            sig = "✓" if b.is_significant else ""
            lines.append(
                f"| {b.edge_min:.0%}-{b.edge_max:.0%} | {b.n_bets} | "
                f"{b.win_rate:.1%} | {b.roi:+.1%} | {b.p_value:.3f} | {sig} |"
            )

        lines.append("")
        lines.append(f"Total: {self.total_bets} bets, {self.overall_win_rate:.1%} win rate, {self.overall_roi:+.1%} ROI")
        return "\n".join(lines)


@dataclass
class EdgeThresholdResult:
    """Result of testing a specific edge threshold."""

    threshold: float
    n_bets: int
    wins: int
    losses: int
    win_rate: float
    roi: float

    # Statistics
    expected_profit: float
    variance: float
    sharpe_ratio: float
    p_value: float
    ci_lower: float
    ci_upper: float

    # Bet quality
    avg_edge: float
    max_edge: float


@dataclass
class EdgeValidationResult:
    """Complete edge validation analysis."""

    model_name: str
    threshold_results: list[EdgeThresholdResult]
    bucket_analysis: EdgeBucketAnalysis

    # Optimal thresholds
    optimal_roi_threshold: float
    optimal_sharpe_threshold: float
    recommended_threshold: float

    # CLV analysis
    avg_clv: Optional[float] = None
    clv_positive_rate: Optional[float] = None

    def summary(self) -> str:
        """Generate comprehensive summary."""
        lines = [
            f"Edge Validation: {self.model_name}",
            "=" * 70,
            "",
            "OPTIMAL THRESHOLDS:",
            f"  Best ROI: {self.optimal_roi_threshold:.1%} edge minimum",
            f"  Best Sharpe: {self.optimal_sharpe_threshold:.1%} edge minimum",
            f"  Recommended: {self.recommended_threshold:.1%} edge minimum",
            "",
            "THRESHOLD ANALYSIS:",
            "| Min Edge | Bets | Win Rate | ROI | Sharpe | p-value |",
            "|----------|------|----------|-----|--------|---------|",
        ]

        for r in self.threshold_results:
            lines.append(
                f"| {r.threshold:.0%} | {r.n_bets} | {r.win_rate:.1%} | "
                f"{r.roi:+.1%} | {r.sharpe_ratio:.2f} | {r.p_value:.3f} |"
            )

        if self.avg_clv is not None:
            lines.append("")
            lines.append("CLV ANALYSIS:")
            lines.append(f"  Average CLV: {self.avg_clv:+.2f} points")
            lines.append(f"  CLV+ Rate: {self.clv_positive_rate:.1%}")

        lines.append("")
        lines.append(self.bucket_analysis.summary())

        return "\n".join(lines)


class EdgeValidator:
    """
    Validates that detected edges are real and profitable.

    Tests:
    - Statistical significance of edge-based performance
    - Optimal edge threshold discovery
    - Edge bucket analysis
    - CLV (Closing Line Value) correlation
    """

    def __init__(
        self,
        stake_per_bet: float = 100.0,
        juice: float = 0.0454,  # -110 juice = 4.54%
        min_bets_per_bucket: int = 20,
    ):
        self.stake_per_bet = stake_per_bet
        self.juice = juice
        self.breakeven_rate = 0.5 + self.juice / 2  # ~52.27% for -110
        self.min_bets_per_bucket = min_bets_per_bucket
        self.logger = logger.bind(component="edge_validator")

    def validate(
        self,
        edges: np.ndarray,
        won: np.ndarray,
        model_name: str = "Model",
        closing_line_values: Optional[np.ndarray] = None,
    ) -> EdgeValidationResult:
        """
        Perform comprehensive edge validation.

        Args:
            edges: Array of edge values (0-1)
            won: Array of bet outcomes (True/False)
            model_name: Name for reporting
            closing_line_values: Optional CLV for each bet

        Returns:
            EdgeValidationResult with full analysis
        """
        edges = np.asarray(edges)
        won = np.asarray(won).astype(bool)

        self.logger.info(f"Validating {len(edges)} bets for {model_name}")

        # Test multiple thresholds
        thresholds = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15]
        threshold_results = [
            self._test_threshold(edges, won, t)
            for t in thresholds
        ]
        threshold_results = [r for r in threshold_results if r is not None]

        # Edge bucket analysis
        bucket_analysis = self._analyze_buckets(edges, won)

        # Find optimal thresholds
        if threshold_results:
            by_roi = max(threshold_results, key=lambda x: x.roi if x.n_bets >= 50 else -999)
            by_sharpe = max(threshold_results, key=lambda x: x.sharpe_ratio if x.n_bets >= 50 else -999)

            # Recommended: balance ROI and sample size
            valid_results = [r for r in threshold_results if r.n_bets >= 100 and r.p_value < 0.10]
            if valid_results:
                recommended = max(valid_results, key=lambda x: x.roi)
            else:
                recommended = by_roi

            optimal_roi_threshold = by_roi.threshold
            optimal_sharpe_threshold = by_sharpe.threshold
            recommended_threshold = recommended.threshold
        else:
            optimal_roi_threshold = 0.03
            optimal_sharpe_threshold = 0.03
            recommended_threshold = 0.03

        # CLV analysis
        avg_clv = None
        clv_positive_rate = None
        if closing_line_values is not None:
            avg_clv = float(np.mean(closing_line_values))
            clv_positive_rate = float(np.mean(closing_line_values > 0))

        return EdgeValidationResult(
            model_name=model_name,
            threshold_results=threshold_results,
            bucket_analysis=bucket_analysis,
            optimal_roi_threshold=optimal_roi_threshold,
            optimal_sharpe_threshold=optimal_sharpe_threshold,
            recommended_threshold=recommended_threshold,
            avg_clv=avg_clv,
            clv_positive_rate=clv_positive_rate,
        )

    def _test_threshold(
        self,
        edges: np.ndarray,
        won: np.ndarray,
        threshold: float,
    ) -> Optional[EdgeThresholdResult]:
        """Test a specific edge threshold."""
        mask = edges >= threshold

        if np.sum(mask) < 10:
            return None

        filtered_won = won[mask]
        filtered_edges = edges[mask]

        n_bets = len(filtered_won)
        wins = int(np.sum(filtered_won))
        losses = n_bets - wins

        win_rate = wins / n_bets
        roi = self._calculate_roi(wins, losses)

        # Expected profit and variance
        expected_profit = n_bets * self.stake_per_bet * roi
        bet_variance = self.stake_per_bet ** 2 * (100/110) ** 2 * win_rate * (1 - win_rate)
        variance = n_bets * bet_variance

        sharpe = expected_profit / np.sqrt(variance) if variance > 0 else 0.0

        # Statistical significance
        result = stats.binomtest(wins, n_bets, self.breakeven_rate, alternative='greater')
        p_value = result.pvalue
        ci = result.proportion_ci(confidence_level=0.95)

        return EdgeThresholdResult(
            threshold=threshold,
            n_bets=n_bets,
            wins=wins,
            losses=losses,
            win_rate=win_rate,
            roi=roi,
            expected_profit=expected_profit,
            variance=variance,
            sharpe_ratio=sharpe,
            p_value=p_value,
            ci_lower=ci.low,
            ci_upper=ci.high,
            avg_edge=float(np.mean(filtered_edges)),
            max_edge=float(np.max(filtered_edges)),
        )

    def _analyze_buckets(
        self,
        edges: np.ndarray,
        won: np.ndarray,
    ) -> EdgeBucketAnalysis:
        """Analyze performance by edge buckets."""
        bucket_edges = [0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10, 0.15, 1.0]
        buckets = []

        for i in range(len(bucket_edges) - 1):
            min_edge = bucket_edges[i]
            max_edge = bucket_edges[i + 1]

            mask = (edges >= min_edge) & (edges < max_edge)
            bucket_won = won[mask]

            if len(bucket_won) < self.min_bets_per_bucket:
                continue

            n_bets = len(bucket_won)
            wins = int(np.sum(bucket_won))
            losses = n_bets - wins
            win_rate = wins / n_bets
            roi = self._calculate_roi(wins, losses)
            total_profit = n_bets * self.stake_per_bet * roi

            # Statistics
            result = stats.binomtest(wins, n_bets, self.breakeven_rate, alternative='greater')
            ci = result.proportion_ci(confidence_level=0.95)

            buckets.append(EdgeBucket(
                edge_min=min_edge,
                edge_max=max_edge,
                n_bets=n_bets,
                wins=wins,
                losses=losses,
                total_profit=total_profit,
                win_rate=win_rate,
                roi=roi,
                p_value=result.pvalue,
                ci_lower=ci.low,
                ci_upper=ci.high,
            ))

        # Find optimal bucket
        if buckets:
            # Consider cumulative from each threshold
            best_roi = -999
            best_threshold = 0.03

            for bucket in buckets:
                # Cumulative from this threshold
                cumulative_mask = edges >= bucket.edge_min
                cumulative_won = won[cumulative_mask]
                if len(cumulative_won) >= 50:
                    cum_wins = int(np.sum(cumulative_won))
                    cum_roi = self._calculate_roi(cum_wins, len(cumulative_won) - cum_wins)
                    if cum_roi > best_roi:
                        best_roi = cum_roi
                        best_threshold = bucket.edge_min

            optimal_min_edge = best_threshold
            optimal_roi = best_roi
        else:
            optimal_min_edge = 0.03
            optimal_roi = 0.0

        # Overall stats
        total_bets = len(edges)
        overall_wins = int(np.sum(won))
        overall_win_rate = overall_wins / total_bets if total_bets > 0 else 0.0
        overall_roi = self._calculate_roi(overall_wins, total_bets - overall_wins)

        return EdgeBucketAnalysis(
            buckets=buckets,
            optimal_min_edge=optimal_min_edge,
            optimal_roi=optimal_roi,
            total_bets=total_bets,
            overall_win_rate=overall_win_rate,
            overall_roi=overall_roi,
        )

    def _calculate_roi(self, wins: int, losses: int) -> float:
        """Calculate ROI for given record at -110 juice."""
        if wins + losses == 0:
            return 0.0
        profit = wins * (100/110) * self.stake_per_bet - losses * self.stake_per_bet
        staked = (wins + losses) * self.stake_per_bet
        return profit / staked


def validate_edges_are_real(
    predicted_edges: np.ndarray,
    actual_outcomes: np.ndarray,
    n_simulations: int = 10000,
) -> dict:
    """
    Test if detected edges are better than random.

    Uses permutation testing to determine if edge-based betting
    outperforms random selection.

    Args:
        predicted_edges: Model's edge predictions
        actual_outcomes: Whether bets won (True/False)
        n_simulations: Number of permutation simulations

    Returns:
        Dict with p-value and analysis
    """
    logger.info(f"Running permutation test with {n_simulations} simulations")

    # Actual performance of edge-based betting
    # Bet on top 20% of edges
    threshold = np.percentile(predicted_edges, 80)
    high_edge_mask = predicted_edges >= threshold

    actual_win_rate = np.mean(actual_outcomes[high_edge_mask])
    actual_n = np.sum(high_edge_mask)

    # Simulate random selection
    simulated_win_rates = []
    for _ in range(n_simulations):
        # Randomly shuffle outcomes
        shuffled = np.random.permutation(actual_outcomes)
        sim_win_rate = np.mean(shuffled[high_edge_mask])
        simulated_win_rates.append(sim_win_rate)

    simulated_win_rates = np.array(simulated_win_rates)

    # P-value: how often does random beat actual?
    p_value = np.mean(simulated_win_rates >= actual_win_rate)

    # Effect size
    effect_size = (actual_win_rate - np.mean(simulated_win_rates)) / np.std(simulated_win_rates)

    return {
        "actual_win_rate": float(actual_win_rate),
        "random_win_rate": float(np.mean(simulated_win_rates)),
        "p_value": float(p_value),
        "effect_size": float(effect_size),
        "n_bets_tested": int(actual_n),
        "is_significant": p_value < 0.05,
        "interpretation": (
            "Edges appear REAL and meaningful" if p_value < 0.05
            else "Edges may be NOISE - not statistically significant"
        ),
    }
