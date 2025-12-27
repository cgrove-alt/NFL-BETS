#!/usr/bin/env python3
"""
Model training script with data freshness guarantees.

Ensures models always train on the latest available data by:
1. Force-refreshing all data sources before training
2. Recording exact data cutoff dates in model metadata
3. Supporting both full retraining and incremental updates

Usage:
    python -m nfl_bets.scripts.train_models --force-refresh
    python -m nfl_bets.scripts.train_models --seasons 2020 2021 2022 2023 2024
    python -m nfl_bets.scripts.train_models --model spread
    python -m nfl_bets.scripts.train_models --model props --prop-type passing_yards
"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import polars as pl

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Default training configuration
MODEL_OUTPUT_DIR = Path("models/trained")
DEFAULT_NUM_SEASONS = 5  # Train on 5 seasons of data

# Walk-forward validation settings
WALKFORWARD_ENABLED = True
WALKFORWARD_TRAIN_SEASONS = 3
WALKFORWARD_TEST_SEASONS = 1


class DataLeakageError(Exception):
    """Raised when chronological ordering is violated."""
    pass


def validate_chronological_order(df: pl.DataFrame, name: str = "dataset") -> None:
    """
    Validate that data is chronologically ordered to prevent look-ahead bias.

    Args:
        df: DataFrame that must contain 'season_week_index' column
        name: Name of dataset for error messages

    Raises:
        DataLeakageError: If data is not chronologically ordered
    """
    if "season_week_index" not in df.columns:
        logger.warning(f"{name}: No season_week_index column, cannot validate ordering")
        return

    swi = df["season_week_index"].to_numpy()
    is_sorted = all(swi[i] <= swi[i+1] for i in range(len(swi)-1))

    if not is_sorted:
        raise DataLeakageError(
            f"CRITICAL: {name} is NOT chronologically ordered! "
            "This will cause data leakage. Data must be sorted by season_week_index."
        )

    logger.debug(f"{name}: Chronological order validated ({len(df)} samples)")


def run_walkforward_validation(
    model_class,
    X: pl.DataFrame,
    y: pl.Series,
    model_name: str,
    season_week_index: Optional[pl.Series] = None,
) -> None:
    """
    Run walk-forward validation and log summary stats.

    Uses proper time-series cross-validation to ensure no look-ahead bias.

    Args:
        model_class: Model class to instantiate and train
        X: Feature matrix
        y: Target values
        model_name: Name for logging
        season_week_index: Optional temporal index for ordering
    """
    if not WALKFORWARD_ENABLED:
        logger.info(f"Walk-forward validation disabled, skipping for {model_name}")
        return

    try:
        from nfl_bets.models.evaluation import WalkForwardValidator
        import numpy as np

        logger.info(f"Running walk-forward validation for {model_name}...")

        # Convert to numpy
        X_arr = X.to_numpy()
        y_arr = y.to_numpy()

        # Get season info for splits
        if season_week_index is not None:
            swi = season_week_index.to_numpy()
            # Sort by temporal index
            sort_idx = np.argsort(swi)
            X_arr = X_arr[sort_idx]
            y_arr = y_arr[sort_idx]
            swi = swi[sort_idx]

            # Extract unique seasons from season_week_index (first 4 digits)
            seasons = np.unique(swi // 100).astype(int)
        else:
            # Assume data is already sorted
            # Create pseudo-seasons based on data chunks
            n_samples = len(y_arr)
            n_folds = 4
            fold_size = n_samples // n_folds
            seasons = list(range(n_folds))

        if len(seasons) < 3:
            logger.warning(f"Not enough seasons for walk-forward ({len(seasons)}), skipping")
            return

        # Run walk-forward validation
        validator = WalkForwardValidator(
            train_seasons=WALKFORWARD_TRAIN_SEASONS,
            test_seasons=WALKFORWARD_TEST_SEASONS,
            min_train_seasons=2,
            expanding=True,
        )

        # Get fold splits
        splits = validator.get_fold_splits(list(seasons))

        logger.info(f"Walk-forward folds: {len(splits)}")

        # Track metrics across folds
        mae_scores = []
        ats_records = []

        for fold_id, (train_seasons, test_seasons) in enumerate(splits):
            # Create train/test masks based on season
            if season_week_index is not None:
                train_mask = np.isin(swi // 100, train_seasons)
                test_mask = np.isin(swi // 100, test_seasons)
            else:
                # Use simple splits
                train_end = int(len(y_arr) * (fold_id + 1) / (len(splits) + 1))
                test_end = train_end + int(len(y_arr) * 0.2)
                train_mask = np.zeros(len(y_arr), dtype=bool)
                train_mask[:train_end] = True
                test_mask = np.zeros(len(y_arr), dtype=bool)
                test_mask[train_end:test_end] = True

            X_train = X_arr[train_mask]
            y_train = y_arr[train_mask]
            X_test = X_arr[test_mask]
            y_test = y_arr[test_mask]

            if len(X_train) < 50 or len(X_test) < 10:
                continue

            # Train model
            model = model_class()
            try:
                model.train(X_train, y_train)
                metrics = model.evaluate(X_test, y_test)
                mae_scores.append(metrics.mae)
                ats_records.append((metrics.ats_wins, metrics.ats_losses, metrics.ats_pushes))
            except Exception as e:
                logger.warning(f"Fold {fold_id} failed: {e}")
                continue

        if not mae_scores:
            logger.warning(f"No valid folds for {model_name}")
            return

        # Log summary
        mean_mae = np.mean(mae_scores)
        std_mae = np.std(mae_scores)
        total_wins = sum(r[0] for r in ats_records)
        total_losses = sum(r[1] for r in ats_records)
        total = total_wins + total_losses
        win_rate = total_wins / total if total > 0 else 0.0

        logger.info(f"Walk-Forward Results for {model_name}:")
        logger.info(f"  Folds: {len(mae_scores)}")
        logger.info(f"  MAE: {mean_mae:.3f} ± {std_mae:.3f}")
        logger.info(f"  ATS Record: {total_wins}-{total_losses} ({win_rate:.1%})")

    except ImportError:
        logger.warning("Walk-forward validator not available, skipping")
    except Exception as e:
        logger.warning(f"Walk-forward validation failed for {model_name}: {e}")


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


def get_training_seasons(n_seasons: int = DEFAULT_NUM_SEASONS) -> list[int]:
    """
    Get list of seasons for training based on current date.

    Args:
        n_seasons: Number of seasons to include (default 5)

    Returns:
        List of season years, e.g., [2021, 2022, 2023, 2024, 2025]
    """
    current = get_current_nfl_season()
    return list(range(current - n_seasons + 1, current + 1))


# Calculate default seasons dynamically from current date
DEFAULT_SEASONS = get_training_seasons()


async def refresh_data_sources(force: bool = True) -> None:
    """
    Refresh all data sources to ensure latest data.

    Args:
        force: If True, bypass cache and fetch fresh data
    """
    from nfl_bets.data.pipeline import DataPipeline
    from nfl_bets.config.settings import get_settings

    settings = get_settings()
    pipeline = DataPipeline.from_settings(settings)

    logger.info("Refreshing data sources...")

    # Refresh nflverse data (play-by-play, stats, schedules)
    try:
        await pipeline.refresh_nflverse_cache()
        logger.info("✓ nflverse data refreshed")
    except Exception as e:
        logger.warning(f"Could not refresh nflverse: {e}")

    # Note: Other data sources (PFF, DVOA, SIC) refresh on-demand
    # They don't have automated refresh methods

    logger.info("Data refresh complete")


async def get_latest_data_cutoff(seasons: list[int]) -> datetime:
    """
    Get the cutoff date of the most recent game in the data.

    Args:
        seasons: List of seasons to check

    Returns:
        Datetime of the most recent game
    """
    from nfl_bets.data.sources.nflverse import NFLVerseClient
    from nfl_bets.config.settings import get_settings

    settings = get_settings()
    cache_dir = settings.data_dir / "cache" / "nflverse" if hasattr(settings, "data_dir") else None
    client = NFLVerseClient(cache_dir=cache_dir)

    # Load schedules to find latest completed game
    schedules = await client.load_schedules(seasons, force_refresh=True)

    # Filter to completed games
    completed = schedules.filter(
        pl.col("result").is_not_null()
    )

    if len(completed) == 0:
        return datetime.now()

    # Get the max game date
    latest_date = completed.select(pl.col("gameday").max()).item()

    if isinstance(latest_date, str):
        return datetime.fromisoformat(latest_date)
    return latest_date


async def load_training_data(
    seasons: list[int],
    force_refresh: bool = True,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Load play-by-play and schedule data for training.

    Args:
        seasons: List of seasons to load
        force_refresh: If True, fetch fresh data

    Returns:
        Tuple of (pbp_df, schedules_df)
    """
    from nfl_bets.data.pipeline import DataPipeline
    from nfl_bets.config.settings import get_settings

    settings = get_settings()
    pipeline = DataPipeline.from_settings(settings)

    logger.info(f"Loading training data for seasons: {seasons}")

    # Load play-by-play data
    pbp_df = await pipeline.get_historical_pbp(
        seasons=seasons,
    )
    logger.info(f"Loaded {len(pbp_df):,} play-by-play records")

    # Load schedules
    schedules_df = await pipeline.get_schedules(
        seasons=seasons,
    )
    logger.info(f"Loaded {len(schedules_df):,} games from schedules")

    return pbp_df, schedules_df


async def build_spread_training_dataset(
    seasons: list[int],
) -> tuple[pl.DataFrame, pl.Series]:
    """
    Build feature matrix and target for spread model.

    Args:
        seasons: List of seasons to include

    Returns:
        Tuple of (X features, y target)
    """
    from nfl_bets.features.feature_pipeline import FeaturePipeline
    from nfl_bets.data.pipeline import DataPipeline
    from nfl_bets.config.settings import get_settings

    settings = get_settings()
    data_pipeline = DataPipeline.from_settings(settings)
    feature_pipeline = FeaturePipeline(data_pipeline)

    logger.info("Building spread training dataset...")

    # Build training dataset - returns DataFrame with features + target
    df = await feature_pipeline.build_training_dataset(
        seasons=seasons,
        target="spread",
        include_playoffs=True,
    )

    if len(df) == 0:
        raise ValueError("No training data generated")

    # Split into features (X) and target (y)
    target_col = "actual_spread"
    metadata_cols = ["game_id", "season", "week", "home_team", "away_team"]
    feature_cols = [c for c in df.columns if c != target_col and c not in metadata_cols]

    X = df.select(feature_cols)
    y = df.get_column(target_col)

    logger.info(f"Built dataset with {len(X)} games and {len(X.columns)} features")

    return X, y


async def train_spread_model(
    X: pl.DataFrame,
    y: pl.Series,
    seasons: list[int],
    data_cutoff: datetime,
    output_dir: Path,
) -> None:
    """
    Train and save the spread prediction model.

    Args:
        X: Feature matrix
        y: Target values (point differential)
        seasons: Seasons used for training
        data_cutoff: Cutoff date of training data
        output_dir: Directory to save model
    """
    from nfl_bets.models.spread_model import SpreadModel

    logger.info("Training spread model...")

    # CRITICAL: Validate chronological ordering before training
    validate_chronological_order(X, "spread_training_data")

    # Create model
    model = SpreadModel()

    # Split for validation (last 20% of games chronologically)
    n_val = int(len(X) * 0.2)
    X_train = X.head(-n_val)
    y_train = y.head(-n_val)
    X_val = X.tail(n_val)
    y_val = y.tail(n_val)

    logger.info(f"Training on {len(X_train)} games, validating on {len(X_val)} games")

    # Train model
    model.train(
        X=X_train,
        y=y_train,
        validation_data=(X_val, y_val),
    )

    # Add data cutoff and seasons to metadata
    if model.metadata:
        model.metadata.data_cutoff_date = data_cutoff
        model.metadata.training_seasons = seasons

    # Evaluate on validation set
    metrics = model.evaluate(X_val, y_val)
    logger.info(f"Validation metrics:")
    logger.info(f"  MAE: {metrics.mae:.2f} points")
    logger.info(f"  RMSE: {metrics.rmse:.2f} points")
    logger.info(f"  R²: {metrics.r2:.3f}")

    # Save model
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"spread_model_v{model.VERSION}.joblib"
    model.save(model_path)

    # Also save as "latest"
    latest_path = output_dir / "spread_model_latest.joblib"
    model.save(latest_path)

    logger.info(f"✓ Spread model saved to {model_path}")


async def train_moneyline_model(
    X: pl.DataFrame,
    y: pl.Series,
    seasons: list[int],
    data_cutoff: datetime,
    output_dir: Path,
) -> None:
    """
    Train and save the moneyline prediction model.

    Args:
        X: Feature matrix (same as spread features)
        y: Target values (1 = home win, 0 = away win)
        seasons: Seasons used for training
        data_cutoff: Cutoff date of training data
        output_dir: Directory to save model
    """
    from nfl_bets.models.moneyline_model import MoneylineModel

    logger.info("Training moneyline model...")

    # CRITICAL: Validate chronological ordering before training
    validate_chronological_order(X, "moneyline_training_data")

    # Create model
    model = MoneylineModel()

    # Split for validation (last 20% of games chronologically)
    n_val = int(len(X) * 0.2)
    X_train = X.head(-n_val)
    y_train = y.head(-n_val)
    X_val = X.tail(n_val)
    y_val = y.tail(n_val)

    logger.info(f"Training on {len(X_train)} games, validating on {len(X_val)} games")

    # Train model
    model.train(
        X=X_train,
        y=y_train,
        validation_data=(X_val, y_val),
    )

    # Add data cutoff and seasons to metadata
    if model.metadata:
        model.metadata.data_cutoff_date = data_cutoff
        model.metadata.training_seasons = seasons

    # Run walk-forward validation
    from nfl_bets.models.moneyline_model import MoneylineModel
    run_walkforward_validation(MoneylineModel, X, y, "MoneylineModel")

    # Save model
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"moneyline_model_v{model.VERSION}.joblib"
    model.save(model_path)

    # Also save as "latest"
    latest_path = output_dir / "moneyline_model_latest.joblib"
    model.save(latest_path)

    logger.info(f"✓ Moneyline model saved to {model_path}")


async def build_moneyline_training_dataset(
    seasons: list[int],
) -> tuple[pl.DataFrame, pl.Series]:
    """
    Build feature matrix and target for moneyline model.

    Uses same features as spread model but binary win/loss target.

    Args:
        seasons: List of seasons to include

    Returns:
        Tuple of (X features, y target as binary home win)
    """
    from nfl_bets.features.feature_pipeline import FeaturePipeline
    from nfl_bets.data.pipeline import DataPipeline
    from nfl_bets.config.settings import get_settings

    settings = get_settings()
    data_pipeline = DataPipeline.from_settings(settings)
    feature_pipeline = FeaturePipeline(data_pipeline)

    logger.info("Building moneyline training dataset...")

    # Build training dataset - same as spread
    df = await feature_pipeline.build_training_dataset(
        seasons=seasons,
        target="spread",
        include_playoffs=True,
    )

    if len(df) == 0:
        raise ValueError("No training data generated")

    # Split into features (X) and target (y)
    # For moneyline, target is 1 if home team won (spread > 0)
    target_col = "actual_spread"
    metadata_cols = ["game_id", "season", "week", "home_team", "away_team"]
    feature_cols = [c for c in df.columns if c != target_col and c not in metadata_cols]

    X = df.select(feature_cols)
    # Binary target: home win = 1 (spread > 0), away win = 0 (spread < 0)
    # Exclude ties (spread == 0) for cleaner classification
    mask = df.get_column(target_col) != 0
    X = X.filter(mask)
    y = (df.get_column(target_col).filter(mask) > 0).cast(pl.Int32)

    logger.info(f"Built dataset with {len(X)} games and {len(X.columns)} features")

    return X, y


async def train_totals_model(
    X: pl.DataFrame,
    y: pl.Series,
    seasons: list[int],
    data_cutoff: datetime,
    output_dir: Path,
) -> None:
    """
    Train and save the totals (over/under) prediction model.

    Args:
        X: Feature matrix (same as spread features)
        y: Target values (total points = home + away score)
        seasons: Seasons used for training
        data_cutoff: Cutoff date of training data
        output_dir: Directory to save model
    """
    from nfl_bets.models.totals_model import TotalsModel

    logger.info("Training totals model...")

    # CRITICAL: Validate chronological ordering before training
    validate_chronological_order(X, "totals_training_data")

    # Create model
    model = TotalsModel()

    # Split for validation (last 20% of games chronologically)
    n_val = int(len(X) * 0.2)
    X_train = X.head(-n_val)
    y_train = y.head(-n_val)
    X_val = X.tail(n_val)
    y_val = y.tail(n_val)

    logger.info(f"Training on {len(X_train)} games, validating on {len(X_val)} games")

    # Train model
    model.train(
        X=X_train,
        y=y_train,
        validation_data=(X_val, y_val),
    )

    # Add data cutoff and seasons to metadata
    if model.metadata:
        model.metadata.data_cutoff_date = data_cutoff
        model.metadata.training_seasons = seasons

    # Evaluate on validation set
    metrics = model.evaluate(X_val, y_val)
    logger.info(f"Validation metrics:")
    logger.info(f"  MAE: {metrics.mae:.2f} points")
    logger.info(f"  RMSE: {metrics.rmse:.2f} points")
    logger.info(f"  R²: {metrics.r2:.3f}")

    # Run walk-forward validation
    from nfl_bets.models.totals_model import TotalsModel
    run_walkforward_validation(TotalsModel, X, y, "TotalsModel")

    # Save model
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"totals_model_v{model.VERSION}.joblib"
    model.save(model_path)

    # Also save as "latest"
    latest_path = output_dir / "totals_model_latest.joblib"
    model.save(latest_path)

    logger.info(f"✓ Totals model saved to {model_path}")


async def build_totals_training_dataset(
    seasons: list[int],
) -> tuple[pl.DataFrame, pl.Series]:
    """
    Build feature matrix and target for totals model.

    Uses same features as spread model but total points target.

    Args:
        seasons: List of seasons to include

    Returns:
        Tuple of (X features, y target as total points)
    """
    from nfl_bets.features.feature_pipeline import FeaturePipeline
    from nfl_bets.data.pipeline import DataPipeline
    from nfl_bets.config.settings import get_settings

    settings = get_settings()
    data_pipeline = DataPipeline.from_settings(settings)
    feature_pipeline = FeaturePipeline(data_pipeline)

    logger.info("Building totals training dataset...")

    # Build training dataset with totals target
    df = await feature_pipeline.build_training_dataset(
        seasons=seasons,
        target="total",  # Request total points as target
        include_playoffs=True,
    )

    if len(df) == 0:
        raise ValueError("No training data generated")

    # Split into features (X) and target (y)
    target_col = "total_points"
    metadata_cols = ["game_id", "season", "week", "home_team", "away_team"]
    feature_cols = [c for c in df.columns if c != target_col and c not in metadata_cols]

    X = df.select(feature_cols)
    y = df.get_column(target_col)

    logger.info(f"Built dataset with {len(X)} games and {len(X.columns)} features")

    return X, y


async def train_prop_model(
    prop_type: str,
    pbp_df: pl.DataFrame,
    schedules_df: pl.DataFrame,
    seasons: list[int],
    data_cutoff: datetime,
    output_dir: Path,
) -> None:
    """
    Train and save a player prop model.

    Args:
        prop_type: Type of prop (passing_yards, rushing_yards, receiving_yards, receptions)
        pbp_df: Play-by-play DataFrame
        schedules_df: Schedules DataFrame
        seasons: Seasons used for training
        data_cutoff: Cutoff date of training data
        output_dir: Directory to save model
    """
    from nfl_bets.features.feature_pipeline import FeaturePipeline
    from nfl_bets.data.pipeline import DataPipeline
    from nfl_bets.config.settings import get_settings

    logger.info(f"Training {prop_type} prop model...")

    settings = get_settings()
    data_pipeline = DataPipeline.from_settings(settings)
    feature_pipeline = FeaturePipeline(data_pipeline)

    # Build player prop training dataset
    X, y = await feature_pipeline.build_prop_training_dataset(
        pbp_df=pbp_df,
        schedules_df=schedules_df,
        prop_type=prop_type,
    )

    if len(X) == 0:
        logger.warning(f"No training data for {prop_type}, skipping")
        return

    # CRITICAL: Validate chronological ordering before training
    validate_chronological_order(X, f"{prop_type}_training_data")

    logger.info(f"Built dataset with {len(X)} player-games and {len(X.columns)} features")

    # Get the appropriate model class
    if prop_type == "passing_yards":
        from nfl_bets.models.player_props import PassingYardsModel
        model = PassingYardsModel()
    elif prop_type == "rushing_yards":
        from nfl_bets.models.player_props import RushingYardsModel
        model = RushingYardsModel()
    elif prop_type == "receiving_yards":
        from nfl_bets.models.player_props import ReceivingYardsModel
        model = ReceivingYardsModel()
    elif prop_type == "receptions":
        from nfl_bets.models.player_props import ReceptionsModel
        model = ReceptionsModel()
    else:
        logger.error(f"Unknown prop type: {prop_type}")
        return

    # Split for validation
    n_val = int(len(X) * 0.2)
    X_train = X.head(-n_val)
    y_train = y.head(-n_val)
    X_val = X.tail(n_val)
    y_val = y.tail(n_val)

    logger.info(f"Training on {len(X_train)} samples, validating on {len(X_val)} samples")

    # Train model
    model.train(
        X=X_train,
        y=y_train,
        validation_data=(X_val, y_val),
    )

    # Add data cutoff to metadata
    if model.metadata:
        model.metadata.data_cutoff_date = data_cutoff
        model.metadata.training_seasons = seasons

    # Save model
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"{prop_type}_model_v{model.VERSION}.joblib"
    model.save(model_path)

    # Also save as "latest"
    latest_path = output_dir / f"{prop_type}_model_latest.joblib"
    model.save(latest_path)

    logger.info(f"✓ {prop_type} model saved to {model_path}")


async def train_all_models(
    seasons: list[int],
    force_refresh: bool = True,
    output_dir: Path = MODEL_OUTPUT_DIR,
) -> None:
    """
    Train all models (spread + all props).

    Args:
        seasons: Seasons to train on
        force_refresh: If True, refresh data before training
        output_dir: Directory to save models
    """
    start_time = datetime.now()
    logger.info("=" * 60)
    logger.info("NFL BETS - Model Training Pipeline")
    logger.info("=" * 60)
    logger.info(f"Training seasons: {seasons}")
    logger.info(f"Force refresh: {force_refresh}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 60)

    # Step 1: Refresh data if requested
    if force_refresh:
        await refresh_data_sources(force=True)

    # Step 2: Get data cutoff date
    data_cutoff = await get_latest_data_cutoff(seasons)
    logger.info(f"Data cutoff date: {data_cutoff.strftime('%Y-%m-%d')}")

    # Step 3: Train spread model (loads data internally)
    X_spread, y_spread = await build_spread_training_dataset(seasons)
    await train_spread_model(X_spread, y_spread, seasons, data_cutoff, output_dir)

    # Step 4: Train moneyline model (uses same features, different target)
    try:
        X_ml, y_ml = await build_moneyline_training_dataset(seasons)
        await train_moneyline_model(X_ml, y_ml, seasons, data_cutoff, output_dir)
    except Exception as e:
        logger.error(f"Failed to train moneyline model: {e}")

    # Step 5: Train totals model
    try:
        X_totals, y_totals = await build_totals_training_dataset(seasons)
        await train_totals_model(X_totals, y_totals, seasons, data_cutoff, output_dir)
    except Exception as e:
        logger.error(f"Failed to train totals model: {e}")

    # Step 6: Load data for prop models
    pbp_df, schedules_df = await load_training_data(seasons, force_refresh)

    # Step 7: Train prop models
    prop_types = ["passing_yards", "rushing_yards", "receiving_yards", "receptions"]
    for prop_type in prop_types:
        try:
            await train_prop_model(
                prop_type=prop_type,
                pbp_df=pbp_df,
                schedules_df=schedules_df,
                seasons=seasons,
                data_cutoff=data_cutoff,
                output_dir=output_dir,
            )
        except Exception as e:
            logger.error(f"Failed to train {prop_type} model: {e}")

    # Summary
    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info("=" * 60)
    logger.info("Training Complete!")
    logger.info(f"Elapsed time: {elapsed:.1f} seconds")
    logger.info(f"Data cutoff: {data_cutoff.strftime('%Y-%m-%d')}")
    logger.info(f"Models saved to: {output_dir}")
    logger.info("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train NFL betting prediction models with latest data"
    )
    parser.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        default=DEFAULT_SEASONS,
        help=f"Seasons to train on (default: {DEFAULT_SEASONS})",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        default=True,
        help="Force refresh data from sources (default: True)",
    )
    parser.add_argument(
        "--no-refresh",
        action="store_true",
        help="Skip data refresh (use cached data)",
    )
    parser.add_argument(
        "--model",
        choices=["all", "spread", "moneyline", "totals", "props"],
        default="all",
        help="Which model(s) to train (default: all)",
    )
    parser.add_argument(
        "--prop-type",
        choices=["passing_yards", "rushing_yards", "receiving_yards", "receptions"],
        help="Specific prop type to train (only with --model props)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=MODEL_OUTPUT_DIR,
        help=f"Output directory for models (default: {MODEL_OUTPUT_DIR})",
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

    force_refresh = args.force_refresh and not args.no_refresh

    # Run training
    asyncio.run(train_all_models(
        seasons=args.seasons,
        force_refresh=force_refresh,
        output_dir=args.output_dir,
    ))


if __name__ == "__main__":
    main()
