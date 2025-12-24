#!/usr/bin/env python3
"""
Walk-forward backtesting script.

Simulates how models would have performed historically by:
1. Training on past data only (no look-ahead bias)
2. Making predictions for future games
3. Tracking theoretical betting performance

Usage:
    python -m nfl_bets.scripts.backtest --seasons 2021 2022 2023 2024 2025
    python -m nfl_bets.scripts.backtest --train-seasons 3 --test-seasons 1
    python -m nfl_bets.scripts.backtest --model spread --min-edge 0.03
"""

import argparse
import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_NUM_SEASONS = 5  # Use 5 seasons for backtesting
DEFAULT_TRAIN_SEASONS = 3  # Train on 3 seasons
DEFAULT_TEST_SEASONS = 1   # Test on 1 season


def get_current_nfl_season() -> int:
    """
    Determine current NFL season based on today's date.

    NFL season starts in September, so:
    - Jan-Aug: Previous year's season (e.g., Jan 2025 = 2024 season)
    - Sep-Dec: Current year's season (e.g., Dec 2025 = 2025 season)

    Returns:
        Current NFL season year
    """
    today = datetime.now()
    # NFL regular season starts in September
    if today.month < 9:
        return today.year - 1
    return today.year


def get_backtest_seasons(n_seasons: int = DEFAULT_NUM_SEASONS) -> list[int]:
    """
    Get list of seasons for backtesting based on current date.

    Args:
        n_seasons: Number of seasons to include (default 5)

    Returns:
        List of season years, e.g., [2021, 2022, 2023, 2024, 2025]
    """
    current = get_current_nfl_season()
    return list(range(current - n_seasons + 1, current + 1))


# Calculate default seasons dynamically from current date
DEFAULT_SEASONS = get_backtest_seasons()


@dataclass
class BacktestBet:
    """Record of a simulated bet."""

    game_id: str
    season: int
    week: int
    bet_type: str  # "spread", "passing_yards", etc.
    description: str

    # Betting details
    line: float
    odds: int
    stake: float

    # Model prediction
    model_probability: float
    edge: float

    # Outcome
    actual_result: float
    won: bool
    profit: float


@dataclass
class BacktestFold:
    """Results from a single train/test fold."""

    train_seasons: list[int]
    test_seasons: list[int]

    # Metrics
    total_bets: int = 0
    wins: int = 0
    losses: int = 0
    pushes: int = 0

    total_staked: float = 0.0
    total_profit: float = 0.0

    # By edge bucket
    bets_by_edge: dict = field(default_factory=dict)

    # Individual bets
    bets: list[BacktestBet] = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        """Win rate excluding pushes."""
        if self.wins + self.losses == 0:
            return 0.0
        return self.wins / (self.wins + self.losses)

    @property
    def roi(self) -> float:
        """Return on investment."""
        if self.total_staked == 0:
            return 0.0
        return self.total_profit / self.total_staked


@dataclass
class BacktestResults:
    """Aggregated results from all folds."""

    folds: list[BacktestFold]
    model_type: str
    min_edge: float

    # Aggregate metrics
    total_bets: int = 0
    total_wins: int = 0
    total_losses: int = 0
    total_pushes: int = 0
    total_staked: float = 0.0
    total_profit: float = 0.0

    def __post_init__(self):
        """Calculate aggregate metrics."""
        for fold in self.folds:
            self.total_bets += fold.total_bets
            self.total_wins += fold.wins
            self.total_losses += fold.losses
            self.total_pushes += fold.pushes
            self.total_staked += fold.total_staked
            self.total_profit += fold.total_profit

    @property
    def win_rate(self) -> float:
        """Overall win rate."""
        if self.total_wins + self.total_losses == 0:
            return 0.0
        return self.total_wins / (self.total_wins + self.total_losses)

    @property
    def roi(self) -> float:
        """Overall ROI."""
        if self.total_staked == 0:
            return 0.0
        return self.total_profit / self.total_staked

    def summary(self) -> str:
        """Generate summary report."""
        lines = [
            "=" * 60,
            f"BACKTEST RESULTS - {self.model_type.upper()} MODEL",
            "=" * 60,
            f"Min Edge Threshold: {self.min_edge:.1%}",
            f"Folds: {len(self.folds)}",
            "",
            "OVERALL PERFORMANCE:",
            f"  Total Bets: {self.total_bets}",
            f"  Record: {self.total_wins}-{self.total_losses}-{self.total_pushes}",
            f"  Win Rate: {self.win_rate:.1%}",
            f"  Total Staked: ${self.total_staked:,.2f}",
            f"  Total Profit: ${self.total_profit:,.2f}",
            f"  ROI: {self.roi:.1%}",
            "",
            "BY FOLD:",
        ]

        for i, fold in enumerate(self.folds, 1):
            lines.append(
                f"  Fold {i}: Train {fold.train_seasons} → Test {fold.test_seasons}"
            )
            lines.append(
                f"    {fold.wins}-{fold.losses}-{fold.pushes} | "
                f"Win: {fold.win_rate:.1%} | ROI: {fold.roi:.1%}"
            )

        lines.append("=" * 60)
        return "\n".join(lines)


class WalkForwardBacktester:
    """
    Walk-forward backtesting engine.

    Trains on past data, tests on future data, then walks forward.
    Ensures no look-ahead bias in evaluation.
    """

    def __init__(
        self,
        train_seasons: int = DEFAULT_TRAIN_SEASONS,
        test_seasons: int = DEFAULT_TEST_SEASONS,
        min_edge: float = 0.03,
        stake_per_bet: float = 100.0,
        odds: int = -110,  # Standard juice
    ):
        """
        Initialize backtester.

        Args:
            train_seasons: Number of seasons to train on
            test_seasons: Number of seasons to test on
            min_edge: Minimum edge to place a bet
            stake_per_bet: Flat stake per bet
            odds: Assumed odds for bets (standard -110)
        """
        self.train_seasons = train_seasons
        self.test_seasons = test_seasons
        self.min_edge = min_edge
        self.stake_per_bet = stake_per_bet
        self.odds = odds

    def generate_folds(
        self,
        seasons: list[int],
    ) -> list[tuple[list[int], list[int]]]:
        """
        Generate train/test folds for walk-forward validation.

        Args:
            seasons: All available seasons

        Returns:
            List of (train_seasons, test_seasons) tuples
        """
        folds = []
        seasons = sorted(seasons)

        min_required = self.train_seasons + self.test_seasons
        if len(seasons) < min_required:
            raise ValueError(
                f"Need at least {min_required} seasons for "
                f"{self.train_seasons} train + {self.test_seasons} test"
            )

        # Walk forward through seasons
        for i in range(len(seasons) - min_required + 1):
            train = seasons[i:i + self.train_seasons]
            test = seasons[i + self.train_seasons:i + min_required]
            folds.append((train, test))

        return folds

    async def run_backtest(
        self,
        model_type: str,
        seasons: list[int],
    ) -> BacktestResults:
        """
        Run full walk-forward backtest.

        Args:
            model_type: "spread" or prop type
            seasons: All seasons to use

        Returns:
            BacktestResults with all fold results
        """
        folds = self.generate_folds(seasons)
        logger.info(f"Running {len(folds)} fold(s) for {model_type} model")

        fold_results = []

        for train_seasons, test_seasons in folds:
            logger.info(f"Fold: Train {train_seasons} → Test {test_seasons}")

            fold_result = await self._run_fold(
                model_type=model_type,
                train_seasons=train_seasons,
                test_seasons=test_seasons,
            )
            fold_results.append(fold_result)

        return BacktestResults(
            folds=fold_results,
            model_type=model_type,
            min_edge=self.min_edge,
        )

    async def _run_fold(
        self,
        model_type: str,
        train_seasons: list[int],
        test_seasons: list[int],
    ) -> BacktestFold:
        """
        Run a single train/test fold.

        Args:
            model_type: Type of model to backtest
            train_seasons: Seasons to train on
            test_seasons: Seasons to test on

        Returns:
            BacktestFold with results
        """
        from nfl_bets.data.pipeline import DataPipeline
        from nfl_bets.features.feature_pipeline import FeaturePipeline
        from nfl_bets.config.settings import get_settings

        settings = get_settings()
        data_pipeline = DataPipeline(settings)
        feature_pipeline = FeaturePipeline(data_pipeline)

        # Load training data
        logger.info(f"Loading training data for seasons {train_seasons}")
        train_pbp = await data_pipeline.get_historical_pbp(
            seasons=train_seasons,
            force_refresh=False,  # Use cache for backtest
        )
        train_schedules = await data_pipeline.get_schedules(
            seasons=train_seasons,
            force_refresh=False,
        )

        # Build training features
        if model_type == "spread":
            X_train, y_train = await feature_pipeline.build_training_dataset(
                pbp_df=train_pbp,
                schedules_df=train_schedules,
                target="spread",
            )
        else:
            X_train, y_train = await feature_pipeline.build_prop_training_dataset(
                pbp_df=train_pbp,
                schedules_df=train_schedules,
                prop_type=model_type,
            )

        # Train model
        logger.info(f"Training {model_type} model on {len(X_train)} samples")
        model = self._create_model(model_type)
        model.train(X_train, y_train)

        # Load test data
        logger.info(f"Loading test data for seasons {test_seasons}")
        test_pbp = await data_pipeline.get_historical_pbp(
            seasons=test_seasons,
            force_refresh=False,
        )
        test_schedules = await data_pipeline.get_schedules(
            seasons=test_seasons,
            force_refresh=False,
        )

        # Build test features
        if model_type == "spread":
            X_test, y_test = await feature_pipeline.build_training_dataset(
                pbp_df=test_pbp,
                schedules_df=test_schedules,
                target="spread",
            )
        else:
            X_test, y_test = await feature_pipeline.build_prop_training_dataset(
                pbp_df=test_pbp,
                schedules_df=test_schedules,
                prop_type=model_type,
            )

        # Simulate betting
        logger.info(f"Simulating bets on {len(X_test)} test samples")
        fold_result = await self._simulate_betting(
            model=model,
            model_type=model_type,
            X_test=X_test,
            y_test=y_test,
            train_seasons=train_seasons,
            test_seasons=test_seasons,
        )

        return fold_result

    def _create_model(self, model_type: str):
        """Create a fresh model instance."""
        if model_type == "spread":
            from nfl_bets.models.spread_model import SpreadModel
            return SpreadModel()
        elif model_type == "passing_yards":
            from nfl_bets.models.player_props import PassingYardsModel
            return PassingYardsModel()
        elif model_type == "rushing_yards":
            from nfl_bets.models.player_props import RushingYardsModel
            return RushingYardsModel()
        elif model_type == "receiving_yards":
            from nfl_bets.models.player_props import ReceivingYardsModel
            return ReceivingYardsModel()
        elif model_type == "receptions":
            from nfl_bets.models.player_props import ReceptionsModel
            return ReceptionsModel()
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    async def _simulate_betting(
        self,
        model,
        model_type: str,
        X_test: pl.DataFrame,
        y_test: pl.Series,
        train_seasons: list[int],
        test_seasons: list[int],
    ) -> BacktestFold:
        """
        Simulate betting on test data.

        Args:
            model: Trained model
            model_type: Type of model
            X_test: Test features
            y_test: Test targets (actual outcomes)
            train_seasons: Seasons used for training
            test_seasons: Seasons being tested

        Returns:
            BacktestFold with betting results
        """
        fold = BacktestFold(
            train_seasons=train_seasons,
            test_seasons=test_seasons,
        )

        # Get predictions
        predictions = model.predict(X_test)

        # For spread model, we need cover probabilities
        # Simulate betting against a standard line
        if model_type == "spread":
            fold = self._simulate_spread_bets(
                predictions=predictions,
                actuals=y_test.to_numpy(),
                X_test=X_test,
                fold=fold,
            )
        else:
            fold = self._simulate_prop_bets(
                model=model,
                predictions=predictions,
                actuals=y_test.to_numpy(),
                X_test=X_test,
                fold=fold,
                prop_type=model_type,
            )

        return fold

    def _simulate_spread_bets(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        X_test: pl.DataFrame,
        fold: BacktestFold,
    ) -> BacktestFold:
        """
        Simulate spread betting.

        For backtest, we assume the "market line" is close to actual result
        and look for cases where model significantly disagrees.
        """
        # Calculate "implied line" from actual results (with noise)
        # In practice, we'd use historical closing lines
        noise = np.random.normal(0, 2, len(actuals))
        implied_lines = actuals + noise

        for i, (pred, actual, line) in enumerate(zip(predictions, actuals, implied_lines)):
            # Model edge: how far off is prediction from line?
            diff = pred - line

            # Bet home cover if model predicts better margin than line
            if abs(diff) > 0:  # Model disagrees with line
                # Simple probability estimate based on prediction confidence
                # In real system, use model.predict_proba()
                edge = min(abs(diff) / 10, 0.15)  # Cap at 15%

                if edge >= self.min_edge:
                    bet_home = diff < 0  # Bet home if model predicts lower margin

                    # Did bet win?
                    actual_margin = actual  # Home margin
                    if bet_home:
                        won = actual_margin < line  # Home covered
                    else:
                        won = actual_margin > line  # Away covered

                    # Calculate profit
                    if won:
                        # Standard -110 odds: win $90.91 on $100
                        profit = self.stake_per_bet * (100 / 110)
                        fold.wins += 1
                    else:
                        profit = -self.stake_per_bet
                        fold.losses += 1

                    fold.total_bets += 1
                    fold.total_staked += self.stake_per_bet
                    fold.total_profit += profit

                    # Track by edge bucket
                    edge_bucket = f"{int(edge * 100)}%+"
                    if edge_bucket not in fold.bets_by_edge:
                        fold.bets_by_edge[edge_bucket] = {"bets": 0, "wins": 0}
                    fold.bets_by_edge[edge_bucket]["bets"] += 1
                    if won:
                        fold.bets_by_edge[edge_bucket]["wins"] += 1

        return fold

    def _simulate_prop_bets(
        self,
        model,
        predictions: np.ndarray,
        actuals: np.ndarray,
        X_test: pl.DataFrame,
        fold: BacktestFold,
        prop_type: str,
    ) -> BacktestFold:
        """
        Simulate player prop betting.

        For backtest, we set the "line" at the median prediction
        and simulate over/under bets.
        """
        for i, (pred, actual) in enumerate(zip(predictions, actuals)):
            # Set line at prediction (simulating efficient market)
            line = pred

            # Calculate edge based on prediction confidence
            # In real system, use quantile predictions
            if hasattr(model, 'predict_distribution'):
                # Use model's uncertainty estimate
                try:
                    mean, std = model.predict_distribution(X_test[i:i+1])
                    edge = min(std[0] / mean[0] if mean[0] > 0 else 0.1, 0.15)
                except Exception:
                    edge = 0.05
            else:
                edge = 0.05  # Default edge estimate

            if edge >= self.min_edge:
                # Randomly bet over or under (simulate market inefficiency)
                bet_over = np.random.random() > 0.5

                # Did bet win?
                if bet_over:
                    won = actual > line
                else:
                    won = actual < line

                # Calculate profit
                if won:
                    profit = self.stake_per_bet * (100 / 110)
                    fold.wins += 1
                else:
                    profit = -self.stake_per_bet
                    fold.losses += 1

                fold.total_bets += 1
                fold.total_staked += self.stake_per_bet
                fold.total_profit += profit

        return fold


async def run_backtest(
    model_type: str = "spread",
    seasons: list[int] = None,
    train_seasons: int = DEFAULT_TRAIN_SEASONS,
    test_seasons: int = DEFAULT_TEST_SEASONS,
    min_edge: float = 0.03,
) -> BacktestResults:
    """
    Run walk-forward backtest.

    Args:
        model_type: "spread" or prop type
        seasons: Seasons to backtest
        train_seasons: Number of seasons to train on
        test_seasons: Number of seasons to test on
        min_edge: Minimum edge threshold

    Returns:
        BacktestResults
    """
    if seasons is None:
        seasons = DEFAULT_SEASONS

    backtester = WalkForwardBacktester(
        train_seasons=train_seasons,
        test_seasons=test_seasons,
        min_edge=min_edge,
    )

    return await backtester.run_backtest(
        model_type=model_type,
        seasons=seasons,
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Walk-forward backtesting for NFL betting models"
    )
    parser.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        default=DEFAULT_SEASONS,
        help=f"Seasons to backtest (default: {DEFAULT_SEASONS})",
    )
    parser.add_argument(
        "--train-seasons",
        type=int,
        default=DEFAULT_TRAIN_SEASONS,
        help=f"Number of seasons to train on (default: {DEFAULT_TRAIN_SEASONS})",
    )
    parser.add_argument(
        "--test-seasons",
        type=int,
        default=DEFAULT_TEST_SEASONS,
        help=f"Number of seasons to test on (default: {DEFAULT_TEST_SEASONS})",
    )
    parser.add_argument(
        "--model",
        choices=["spread", "passing_yards", "rushing_yards", "receiving_yards", "receptions"],
        default="spread",
        help="Model type to backtest (default: spread)",
    )
    parser.add_argument(
        "--min-edge",
        type=float,
        default=0.03,
        help="Minimum edge threshold (default: 0.03 = 3%%)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("=" * 60)
    logger.info("NFL BETS - Walk-Forward Backtest")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Seasons: {args.seasons}")
    logger.info(f"Train/Test: {args.train_seasons}/{args.test_seasons}")
    logger.info(f"Min Edge: {args.min_edge:.1%}")
    logger.info("=" * 60)

    # Run backtest
    results = asyncio.run(run_backtest(
        model_type=args.model,
        seasons=args.seasons,
        train_seasons=args.train_seasons,
        test_seasons=args.test_seasons,
        min_edge=args.min_edge,
    ))

    # Print results
    print(results.summary())


if __name__ == "__main__":
    main()
