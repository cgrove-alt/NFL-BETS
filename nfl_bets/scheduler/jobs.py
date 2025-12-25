"""
Background job definitions for the scheduler.

Each job is an async function that performs a specific task:
- poll_odds: Fetch odds and detect value bets
- check_model_refresh: Verify model freshness
- sync_nflverse: Refresh nflverse cache
- health_check: Monitor data source health
"""

import logging
from datetime import datetime
from typing import Any, Optional

logger = logging.getLogger(__name__)


async def poll_odds(
    pipeline: Any,
    feature_pipeline: Any,
    value_detector: Any,
    bankroll_manager: Any,
) -> list:
    """
    Poll Odds API for latest lines and run value detection.

    This is the core polling job that:
    1. Fetches latest odds from The Odds API
    2. Gets upcoming games with enriched data
    3. Builds features for each game
    4. Runs value detection to find +EV opportunities
    5. Calculates recommended stakes

    Args:
        pipeline: DataPipeline instance
        feature_pipeline: FeaturePipeline instance
        value_detector: ValueDetector instance
        bankroll_manager: BankrollManager instance

    Returns:
        List of ValueBet opportunities found
    """
    from nfl_bets.betting.kelly_calculator import KellyCalculator

    logger.debug("Starting odds poll...")
    start_time = datetime.now()

    try:
        # 1. Get upcoming games
        games = await pipeline.get_upcoming_games()
        if not games:
            logger.debug("No upcoming games found")
            return []

        logger.debug(f"Found {len(games)} upcoming games")

        # 2. Fetch latest odds
        odds_data = await pipeline.get_game_odds()
        if not odds_data:
            logger.warning("No odds data available")
            return []

        logger.debug(f"Fetched odds for {len(odds_data)} games")

        # 3. Build features for each game
        features = {}
        for game in games:
            game_id = game.get("game_id")
            if not game_id:
                continue

            try:
                game_features = await feature_pipeline.build_spread_features_for_game(
                    game_id=game_id,
                    home_team=game.get("home_team"),
                    away_team=game.get("away_team"),
                )
                features[game_id] = game_features
            except Exception as e:
                logger.debug(f"Could not build features for {game_id}: {e}")

        # 4. Run value detection
        result = value_detector.scan_spreads(
            games=games,
            odds_data=odds_data,
            features=features,
        )

        value_bets = result.value_bets

        # 5. Calculate recommended stakes
        kelly = KellyCalculator()
        for bet in value_bets:
            stake = kelly.calculate_stake(
                bankroll=bankroll_manager.current_bankroll,
                win_probability=bet.model_probability,
                odds=bet.odds,
            )
            bet.recommended_stake = stake.recommended_stake

        elapsed = (datetime.now() - start_time).total_seconds()
        logger.debug(f"Odds poll complete in {elapsed:.1f}s, found {len(value_bets)} value bets")

        return value_bets

    except Exception as e:
        logger.error(f"Odds polling failed: {e}")
        raise


async def check_model_refresh(model_manager: Any) -> dict:
    """
    Check model staleness and log warnings if models need retraining.

    Args:
        model_manager: ModelManager instance

    Returns:
        Dict with staleness status for each model
    """
    logger.debug("Checking model freshness...")

    try:
        all_fresh, stale_models = model_manager.check_all_models_fresh()

        if all_fresh:
            logger.info("All models are fresh")
        else:
            logger.warning(f"Stale models detected: {stale_models}")
            logger.warning(
                "Run 'python -m nfl_bets.scripts.train_models --force-refresh' to retrain"
            )

        # Get detailed info for each model
        model_info = {}
        for model_type in ["spread", "passing_yards", "rushing_yards", "receiving_yards"]:
            try:
                info = model_manager.get_model_info(model_type)
                model_info[model_type] = info
            except Exception as e:
                model_info[model_type] = {"error": str(e), "is_stale": True}

        return {
            "all_fresh": all_fresh,
            "stale_models": stale_models,
            "models": model_info,
            "checked_at": datetime.now(),
        }

    except Exception as e:
        logger.error(f"Model refresh check failed: {e}")
        raise


async def sync_nflverse(pipeline: Any) -> dict:
    """
    Nightly sync of nflverse play-by-play and schedule data.

    Forces a refresh of the nflverse cache to ensure
    we have the latest game results and statistics.

    Args:
        pipeline: DataPipeline instance

    Returns:
        Dict with sync status
    """
    logger.info("Starting nflverse data sync...")
    start_time = datetime.now()

    try:
        await pipeline.refresh_nflverse_cache()

        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"nflverse sync complete in {elapsed:.1f}s")

        return {
            "status": "success",
            "elapsed_seconds": elapsed,
            "synced_at": datetime.now(),
        }

    except Exception as e:
        logger.error(f"nflverse sync failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "synced_at": datetime.now(),
        }


async def health_check(pipeline: Any) -> Any:
    """
    Periodic health monitoring of all data sources.

    Checks connectivity and availability of:
    - nflverse (local cache + remote)
    - The Odds API
    - PFF (if configured)
    - FTN/DVOA (if configured)
    - SIC Score (if configured)

    Args:
        pipeline: DataPipeline instance

    Returns:
        PipelineHealth with status for each source
    """
    logger.debug("Running health check...")

    try:
        health = await pipeline.health_check()

        # Log summary
        healthy_count = sum(1 for v in health.source_status.values() if v)
        total_count = len(health.source_status)
        logger.debug(f"Health check: {healthy_count}/{total_count} sources healthy")

        # Log any issues
        for source, is_healthy in health.source_status.items():
            if not is_healthy:
                logger.warning(f"Data source unhealthy: {source}")

        return health

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise


async def trigger_model_retrain() -> dict:
    """
    Trigger model retraining if models are stale.

    This is a manual trigger that runs the training script
    asynchronously. Use with caution as training can take
    several minutes.

    Returns:
        Dict with retrain status
    """
    import subprocess
    import sys

    logger.info("Triggering model retrain...")

    try:
        # Run training script in subprocess
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "nfl_bets.scripts.train_models",
                "--force-refresh",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        stdout, stderr = process.communicate(timeout=1800)  # 30 min timeout

        if process.returncode == 0:
            logger.info("Model retraining completed successfully")
            return {
                "status": "success",
                "message": "Models retrained successfully",
                "retrained_at": datetime.now(),
            }
        else:
            error_msg = stderr.decode() if stderr else "Unknown error"
            logger.error(f"Model retraining failed: {error_msg}")
            return {
                "status": "error",
                "error": error_msg,
                "retrained_at": datetime.now(),
            }

    except subprocess.TimeoutExpired:
        process.kill()
        logger.error("Model retraining timed out")
        return {
            "status": "error",
            "error": "Training timed out after 30 minutes",
            "retrained_at": datetime.now(),
        }
    except Exception as e:
        logger.error(f"Model retraining failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "retrained_at": datetime.now(),
        }
