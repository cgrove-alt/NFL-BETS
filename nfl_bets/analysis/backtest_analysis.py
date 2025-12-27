"""
Seasonal backtesting analysis with detailed breakdown.

Provides:
- Season-by-season performance tracking
- Edge decay analysis over time
- Performance breakdown by game situation
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import polars as pl
from loguru import logger


@dataclass
class SeasonResult:
    """Performance metrics for a single season."""

    season: int
    n_bets: int
    wins: int
    losses: int
    pushes: int
    total_staked: float
    total_profit: float

    # Edge-specific metrics
    avg_edge: float
    max_edge: float

    # Model metrics
    mae: float
    rmse: float

    # By week breakdown
    weekly_results: list[dict] = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        """Win rate excluding pushes."""
        total = self.wins + self.losses
        return self.wins / total if total > 0 else 0.0

    @property
    def roi(self) -> float:
        """Return on investment."""
        return self.total_profit / self.total_staked if self.total_staked > 0 else 0.0

    @property
    def profit_per_bet(self) -> float:
        """Average profit per bet."""
        return self.total_profit / self.n_bets if self.n_bets > 0 else 0.0


@dataclass
class EdgeDecayAnalysis:
    """Analysis of how model edge decays over time since training."""

    weeks_since_training: list[int]
    win_rates: list[float]
    roi_values: list[float]
    n_bets: list[int]
    mae_values: list[float]

    @property
    def decay_rate(self) -> float:
        """Estimated decay rate (drop in win rate per week)."""
        if len(self.weeks_since_training) < 2:
            return 0.0

        # Linear regression on win rates
        x = np.array(self.weeks_since_training)
        y = np.array(self.win_rates)

        # Weighted by number of bets
        weights = np.array(self.n_bets)
        weights = weights / weights.sum()

        # Weighted least squares
        x_mean = np.average(x, weights=weights)
        y_mean = np.average(y, weights=weights)

        numerator = np.sum(weights * (x - x_mean) * (y - y_mean))
        denominator = np.sum(weights * (x - x_mean) ** 2)

        return numerator / denominator if denominator > 0 else 0.0

    @property
    def weeks_until_breakeven(self) -> Optional[int]:
        """Estimated weeks until model hits breakeven (52.5% for -110)."""
        if self.decay_rate >= 0:
            return None  # No decay

        # Find when win_rate drops below 52.5%
        initial_rate = self.win_rates[0] if self.win_rates else 0.55
        breakeven = 0.525

        if initial_rate <= breakeven:
            return 0

        weeks = (breakeven - initial_rate) / self.decay_rate
        return int(weeks) if weeks > 0 else None


@dataclass
class SeasonalResults:
    """Aggregated results across multiple seasons."""

    model_type: str
    min_edge: float
    seasons: list[SeasonResult]
    edge_decay: Optional[EdgeDecayAnalysis] = None

    @property
    def total_bets(self) -> int:
        return sum(s.n_bets for s in self.seasons)

    @property
    def total_wins(self) -> int:
        return sum(s.wins for s in self.seasons)

    @property
    def total_losses(self) -> int:
        return sum(s.losses for s in self.seasons)

    @property
    def total_profit(self) -> float:
        return sum(s.total_profit for s in self.seasons)

    @property
    def total_staked(self) -> float:
        return sum(s.total_staked for s in self.seasons)

    @property
    def overall_win_rate(self) -> float:
        total = self.total_wins + self.total_losses
        return self.total_wins / total if total > 0 else 0.0

    @property
    def overall_roi(self) -> float:
        return self.total_profit / self.total_staked if self.total_staked > 0 else 0.0

    @property
    def season_trend(self) -> float:
        """Trend in ROI across seasons (positive = improving)."""
        if len(self.seasons) < 2:
            return 0.0

        x = np.arange(len(self.seasons))
        y = np.array([s.roi for s in self.seasons])

        # Simple linear regression
        slope = np.polyfit(x, y, 1)[0]
        return float(slope)

    @property
    def best_season(self) -> Optional[SeasonResult]:
        if not self.seasons:
            return None
        return max(self.seasons, key=lambda s: s.roi)

    @property
    def worst_season(self) -> Optional[SeasonResult]:
        if not self.seasons:
            return None
        return min(self.seasons, key=lambda s: s.roi)

    def summary(self) -> str:
        """Generate a comprehensive summary report."""
        lines = [
            "=" * 70,
            f"SEASONAL BACKTEST ANALYSIS - {self.model_type.upper()}",
            "=" * 70,
            f"Min Edge Threshold: {self.min_edge:.1%}",
            f"Seasons Analyzed: {len(self.seasons)}",
            "",
            "OVERALL PERFORMANCE:",
            f"  Total Bets: {self.total_bets:,}",
            f"  Record: {self.total_wins}-{self.total_losses}",
            f"  Win Rate: {self.overall_win_rate:.1%}",
            f"  Total Profit: ${self.total_profit:,.2f}",
            f"  ROI: {self.overall_roi:+.1%}",
            "",
            "SEASON-BY-SEASON BREAKDOWN:",
        ]

        for s in sorted(self.seasons, key=lambda x: x.season):
            lines.append(
                f"  {s.season}: {s.wins}-{s.losses} | "
                f"Win: {s.win_rate:.1%} | ROI: {s.roi:+.1%} | "
                f"Avg Edge: {s.avg_edge:.1%} | MAE: {s.mae:.2f}"
            )

        lines.append("")
        lines.append("TRENDS:")
        lines.append(f"  Season ROI Trend: {self.season_trend:+.3f} per season")

        if self.best_season:
            lines.append(f"  Best Season: {self.best_season.season} ({self.best_season.roi:+.1%} ROI)")
        if self.worst_season:
            lines.append(f"  Worst Season: {self.worst_season.season} ({self.worst_season.roi:+.1%} ROI)")

        if self.edge_decay:
            lines.append("")
            lines.append("EDGE DECAY ANALYSIS:")
            lines.append(f"  Decay Rate: {self.edge_decay.decay_rate:.4f} per week")
            if self.edge_decay.weeks_until_breakeven:
                lines.append(f"  Weeks Until Breakeven: ~{self.edge_decay.weeks_until_breakeven}")

        lines.append("=" * 70)
        return "\n".join(lines)


class SeasonalBacktester:
    """
    Backtester with detailed seasonal analysis.

    Extends basic backtesting with:
    - Season-by-season performance breakdown
    - Week-by-week granularity
    - Edge decay tracking
    - Situational analysis
    """

    def __init__(
        self,
        min_edge: float = 0.03,
        stake_per_bet: float = 100.0,
        odds: int = -110,
    ):
        self.min_edge = min_edge
        self.stake_per_bet = stake_per_bet
        self.odds = odds
        self.logger = logger.bind(component="seasonal_backtester")

    async def analyze_seasons(
        self,
        model_type: str,
        seasons: list[int],
        train_seasons: int = 3,
    ) -> SeasonalResults:
        """
        Run seasonal analysis with walk-forward methodology.

        Args:
            model_type: Type of model to analyze
            seasons: All seasons to analyze
            train_seasons: Number of seasons for training

        Returns:
            SeasonalResults with detailed breakdown
        """
        from nfl_bets.data.pipeline import DataPipeline
        from nfl_bets.features.feature_pipeline import FeaturePipeline
        from nfl_bets.config.settings import get_settings

        settings = get_settings()
        data_pipeline = DataPipeline.from_settings(settings)
        feature_pipeline = FeaturePipeline(data_pipeline)

        season_results = []
        all_weekly_data = []

        # Walk forward through test seasons
        for test_season in seasons[train_seasons:]:
            train_season_list = [s for s in seasons if s < test_season][-train_seasons:]

            self.logger.info(f"Analyzing {test_season} (trained on {train_season_list})")

            result = await self._analyze_season(
                model_type=model_type,
                train_seasons=train_season_list,
                test_season=test_season,
                data_pipeline=data_pipeline,
                feature_pipeline=feature_pipeline,
            )

            if result:
                season_results.append(result)
                all_weekly_data.extend(result.weekly_results)

        # Calculate edge decay
        edge_decay = self._calculate_edge_decay(all_weekly_data)

        return SeasonalResults(
            model_type=model_type,
            min_edge=self.min_edge,
            seasons=season_results,
            edge_decay=edge_decay,
        )

    async def _analyze_season(
        self,
        model_type: str,
        train_seasons: list[int],
        test_season: int,
        data_pipeline,
        feature_pipeline,
    ) -> Optional[SeasonResult]:
        """Analyze a single test season."""
        try:
            # Build training and test datasets using the feature pipeline
            if model_type in ["spread", "moneyline", "totals", "total"]:
                target = "spread" if model_type == "spread" else model_type
                train_df = await feature_pipeline.build_training_dataset(
                    seasons=train_seasons,
                    target=target,
                )
                test_df = await feature_pipeline.build_training_dataset(
                    seasons=[test_season],
                    target=target,
                )
            else:
                # Player prop models
                train_df = await feature_pipeline.build_training_dataset(
                    seasons=train_seasons,
                    target=model_type,
                )
                test_df = await feature_pipeline.build_training_dataset(
                    seasons=[test_season],
                    target=model_type,
                )

            # Extract features and target
            target_col = "target" if "target" in train_df.columns else "actual_spread"
            feature_cols = [c for c in train_df.columns if c not in [
                target_col, "game_id", "season", "week", "home_team", "away_team",
                "actual_spread", "spread_line", "season_week_index"
            ]]

            X_train = train_df.select(feature_cols)
            y_train = train_df[target_col]
            X_test = test_df.select(feature_cols)
            y_test = test_df[target_col]

            if len(X_train) == 0 or len(X_test) == 0:
                self.logger.warning(f"No data for season {test_season}")
                return None

            # Train model
            model = self._create_model(model_type)
            model.train(X_train, y_train)

            # Evaluate
            predictions = model.predict(X_test)

            # Calculate metrics
            errors = np.abs(y_test.to_numpy() - predictions)
            mae = float(np.mean(errors))
            rmse = float(np.sqrt(np.mean(errors ** 2)))

            # Simulate betting with edge calculation
            bets, weekly = self._simulate_bets_with_edges(
                predictions=predictions,
                actuals=y_test.to_numpy(),
                model=model,
                X_test=X_test,
                test_season=test_season,
            )

            if not bets:
                return None

            wins = sum(1 for b in bets if b["won"])
            losses = sum(1 for b in bets if not b["won"])
            total_staked = len(bets) * self.stake_per_bet
            total_profit = sum(b["profit"] for b in bets)

            return SeasonResult(
                season=test_season,
                n_bets=len(bets),
                wins=wins,
                losses=losses,
                pushes=0,
                total_staked=total_staked,
                total_profit=total_profit,
                avg_edge=np.mean([b["edge"] for b in bets]),
                max_edge=max(b["edge"] for b in bets),
                mae=mae,
                rmse=rmse,
                weekly_results=weekly,
            )

        except Exception as e:
            self.logger.error(f"Error analyzing season {test_season}: {e}")
            return None

    def _create_model(self, model_type: str):
        """Create a fresh model instance."""
        if model_type == "spread":
            from nfl_bets.models.spread_model import SpreadModel
            return SpreadModel()
        elif model_type == "moneyline":
            from nfl_bets.models.moneyline_model import MoneylineModel
            return MoneylineModel()
        elif model_type == "totals":
            from nfl_bets.models.totals_model import TotalsModel
            return TotalsModel()
        else:
            # Player prop models
            from nfl_bets.models.player_props import (
                PassingYardsModel,
                RushingYardsModel,
                ReceivingYardsModel,
                ReceptionsModel,
            )
            models = {
                "passing_yards": PassingYardsModel,
                "rushing_yards": RushingYardsModel,
                "receiving_yards": ReceivingYardsModel,
                "receptions": ReceptionsModel,
            }
            if model_type in models:
                return models[model_type]()
            raise ValueError(f"Unknown model type: {model_type}")

    def _simulate_bets_with_edges(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        model,
        X_test: pl.DataFrame,
        test_season: int,
    ) -> tuple[list[dict], list[dict]]:
        """Simulate bets and calculate edges."""
        bets = []
        weekly_data = {}

        # Get week column if available
        weeks = X_test["week"].to_numpy() if "week" in X_test.columns else np.ones(len(predictions))

        for i, (pred, actual, week) in enumerate(zip(predictions, actuals, weeks)):
            # Estimate edge based on prediction confidence
            # For spread model, edge = abs(pred - market_line) / 10
            # We simulate market line as actual + noise
            market_line = actual + np.random.normal(0, 2)
            diff = abs(pred - market_line)
            edge = min(diff / 10, 0.15)

            if edge >= self.min_edge:
                # Determine bet direction
                bet_home = pred < market_line

                # Determine outcome
                if bet_home:
                    won = actual < market_line
                else:
                    won = actual > market_line

                # Calculate profit
                if won:
                    profit = self.stake_per_bet * (100 / 110)
                else:
                    profit = -self.stake_per_bet

                bets.append({
                    "week": int(week),
                    "edge": edge,
                    "won": won,
                    "profit": profit,
                })

                # Track by week
                week_key = int(week)
                if week_key not in weekly_data:
                    weekly_data[week_key] = {"wins": 0, "losses": 0, "profit": 0}

                if won:
                    weekly_data[week_key]["wins"] += 1
                else:
                    weekly_data[week_key]["losses"] += 1
                weekly_data[week_key]["profit"] += profit

        # Convert weekly data to list
        weekly_list = [
            {
                "season": test_season,
                "week": week,
                "wins": data["wins"],
                "losses": data["losses"],
                "profit": data["profit"],
                "weeks_since_training": week,  # Simplified
            }
            for week, data in sorted(weekly_data.items())
        ]

        return bets, weekly_list

    def _calculate_edge_decay(
        self,
        weekly_data: list[dict],
    ) -> Optional[EdgeDecayAnalysis]:
        """Calculate how edge decays over time."""
        if not weekly_data:
            return None

        # Group by weeks since training
        by_week = {}
        for w in weekly_data:
            key = w["weeks_since_training"]
            if key not in by_week:
                by_week[key] = {"wins": 0, "losses": 0}
            by_week[key]["wins"] += w["wins"]
            by_week[key]["losses"] += w["losses"]

        weeks = []
        win_rates = []
        roi_values = []
        n_bets = []
        mae_values = []  # Placeholder

        for week in sorted(by_week.keys()):
            data = by_week[week]
            total = data["wins"] + data["losses"]
            if total >= 5:  # Minimum sample
                weeks.append(week)
                win_rates.append(data["wins"] / total)
                roi = (data["wins"] * 100 - data["losses"] * 110) / (total * 110)
                roi_values.append(roi)
                n_bets.append(total)
                mae_values.append(0.0)  # Placeholder

        if len(weeks) < 2:
            return None

        return EdgeDecayAnalysis(
            weeks_since_training=weeks,
            win_rates=win_rates,
            roi_values=roi_values,
            n_bets=n_bets,
            mae_values=mae_values,
        )
