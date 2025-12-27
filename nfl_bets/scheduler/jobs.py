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

import polars as pl

logger = logging.getLogger(__name__)

# Team name normalization for Odds API full names -> nflverse abbreviations
TEAM_NAME_MAP = {
    "Arizona Cardinals": "ARI",
    "Atlanta Falcons": "ATL",
    "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR",
    "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN",
    "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN",
    "Detroit Lions": "DET",
    "Green Bay Packers": "GB",
    "Houston Texans": "HOU",
    "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV",
    "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LAR",
    "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN",
    "New England Patriots": "NE",
    "New Orleans Saints": "NO",
    "New York Giants": "NYG",
    "New York Jets": "NYJ",
    "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA",
    "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN",
    "Washington Commanders": "WAS",
}


def _transform_odds_data(raw_odds: list[dict], games: list[dict]) -> list[dict]:
    """
    Transform raw Odds API response to format expected by ValueDetector.

    The Odds API returns:
    {
        "id": "abc123",
        "home_team": "Kansas City Chiefs",
        "away_team": "Denver Broncos",
        "bookmakers": [
            {
                "key": "draftkings",
                "markets": [
                    {
                        "key": "spreads",
                        "outcomes": [
                            {"name": "Kansas City Chiefs", "point": -3.5, "price": -110},
                            {"name": "Denver Broncos", "point": 3.5, "price": -110}
                        ]
                    }
                ]
            }
        ]
    }

    ValueDetector expects:
    {
        "game_id": "2024_01_KC_DEN",
        "bookmaker": "draftkings",
        "home_spread": -3.5,
        "home_odds": -110,
        "away_odds": -110
    }

    Args:
        raw_odds: List of raw API responses from The Odds API
        games: List of game dicts with game_id, home_team, away_team

    Returns:
        List of transformed odds dicts for ValueDetector
    """
    # Build lookup from team names to game_id
    # Key: (home_team_normalized, away_team_normalized) -> game_id
    team_to_game = {}
    for game in games:
        home = game.get("home_team", "").upper()
        away = game.get("away_team", "").upper()
        game_id = game.get("game_id")
        if home and away and game_id:
            team_to_game[(home, away)] = game_id

    def normalize_team(name: str) -> str:
        """Normalize team name to abbreviation."""
        return TEAM_NAME_MAP.get(name, name.upper())

    transformed = []

    for odds_event in raw_odds:
        home_team_raw = odds_event.get("home_team", "")
        away_team_raw = odds_event.get("away_team", "")

        home_team = normalize_team(home_team_raw)
        away_team = normalize_team(away_team_raw)

        # Find matching game_id
        game_id = team_to_game.get((home_team, away_team))
        if not game_id:
            # Try with normalized names
            for (h, a), gid in team_to_game.items():
                if h == home_team and a == away_team:
                    game_id = gid
                    break

        if not game_id:
            logger.debug(f"No game_id match for {home_team} vs {away_team}")
            continue

        # Process each bookmaker
        for bookmaker in odds_event.get("bookmakers", []):
            book_key = bookmaker.get("key", "unknown")

            # Find spreads market
            for market in bookmaker.get("markets", []):
                if market.get("key") != "spreads":
                    continue

                outcomes = market.get("outcomes", [])
                if len(outcomes) < 2:
                    continue

                # Parse home and away outcomes
                home_spread = None
                home_odds = None
                away_odds = None

                for outcome in outcomes:
                    outcome_team = normalize_team(outcome.get("name", ""))
                    point = outcome.get("point")
                    price = outcome.get("price")

                    if outcome_team == home_team:
                        home_spread = point
                        home_odds = price
                    elif outcome_team == away_team:
                        away_odds = price

                # Only add if we have all required data
                if home_spread is not None and home_odds is not None and away_odds is not None:
                    transformed.append({
                        "game_id": game_id,
                        "bookmaker": book_key,
                        "home_spread": home_spread,
                        "home_odds": home_odds,
                        "away_odds": away_odds,
                    })

    logger.info(f"Transformed {len(transformed)} odds entries from {len(raw_odds)} API events")
    return transformed


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

    logger.info("=== STARTING ODDS POLL ===")
    start_time = datetime.now()

    try:
        # 1. Get upcoming games (returns Polars DataFrame)
        games_df = await pipeline.get_upcoming_games()
        if games_df is None or len(games_df) == 0:
            logger.warning("POLL FAILED: No upcoming games found from nflverse")
            return []

        # Convert DataFrame to list of dicts for iteration
        games = games_df.to_dicts()
        logger.info(f"Step 1: Found {len(games)} upcoming games")

        # Log sample game data for debugging
        if games:
            sample = games[0]
            logger.info(f"Sample game: {sample.get('game_id')} - {sample.get('away_team')} @ {sample.get('home_team')}, season={sample.get('season')}, week={sample.get('week')}")

        # 2. Fetch latest odds (raw API format)
        raw_odds = await pipeline.get_game_odds()
        if not raw_odds:
            logger.warning("POLL FAILED: No odds data from Odds API - check API key and connectivity")
            return []

        logger.info(f"Step 2: Fetched odds for {len(raw_odds)} games from Odds API")

        # Log sample raw odds for debugging
        if raw_odds:
            sample = raw_odds[0]
            logger.info(f"Sample raw odds: {sample.get('home_team')} vs {sample.get('away_team')}, bookmakers={len(sample.get('bookmakers', []))}")

        # Transform to format expected by ValueDetector
        odds_data = _transform_odds_data(raw_odds, games)
        logger.info(f"Step 3: Transformed {len(odds_data)} odds entries (from {len(raw_odds)} API events)")

        if not odds_data:
            logger.warning("POLL FAILED: No odds matched to games after transformation")
            logger.warning("This usually means team names from Odds API don't match nflverse team abbreviations")
            # Log the teams for debugging
            api_teams = set()
            for o in raw_odds:
                api_teams.add(o.get('home_team', ''))
                api_teams.add(o.get('away_team', ''))
            game_teams = set()
            for g in games:
                game_teams.add(g.get('home_team', ''))
                game_teams.add(g.get('away_team', ''))
            logger.warning(f"Odds API teams (sample): {list(api_teams)[:6]}")
            logger.warning(f"Game teams: {list(game_teams)[:6]}")
            return []

        # 3. Build features for each game
        # Use the season from game data - nflverse labels by season start year
        sample_season = games[0].get("season") if games else None
        pbp_season = int(sample_season) if sample_season else 2025

        # Fetch PBP data with fallback to previous seasons
        pbp_df = None
        seasons_to_try = [pbp_season, pbp_season - 1, 2024, 2023]  # Try multiple seasons
        for season_try in seasons_to_try:
            try:
                logger.info(f"Trying to fetch PBP data for season {season_try}")
                pbp_df = await pipeline.get_historical_pbp([season_try])
                if pbp_df is not None and len(pbp_df) > 0:
                    logger.info(f"Successfully loaded PBP data from season {season_try} ({len(pbp_df)} rows)")
                    pbp_season = season_try  # Update to actual season used
                    break
                else:
                    logger.warning(f"No PBP data for season {season_try}, trying next...")
            except Exception as e:
                logger.warning(f"PBP fetch failed for season {season_try}: {e}")

        if pbp_df is None or len(pbp_df) == 0:
            logger.error(f"CRITICAL: No PBP data available for any season {seasons_to_try}")
            return []

        # Fetch real schedules from nflverse (not just games list)
        logger.info(f"Fetching schedules for season {pbp_season}")
        try:
            schedules_df = await pipeline.get_schedules([pbp_season])
            if schedules_df is None or len(schedules_df) == 0:
                logger.warning(f"No schedules from nflverse, using games list as fallback")
                schedules_df = pl.DataFrame(games)
            else:
                logger.info(f"Loaded {len(schedules_df)} schedule entries from nflverse")
        except Exception as e:
            logger.warning(f"Schedules fetch failed: {e}, using games list as fallback")
            schedules_df = pl.DataFrame(games)

        features = {}
        feature_errors = []
        for game in games:
            game_id = game.get("game_id")
            if not game_id:
                continue

            # Extract season and week from game data
            season = game.get("season")
            week = game.get("week")

            if not season or not week:
                feature_errors.append(f"{game_id}: missing season/week")
                continue

            try:
                game_features = await feature_pipeline.build_spread_features(
                    game_id=game_id,
                    home_team=game.get("home_team"),
                    away_team=game.get("away_team"),
                    season=pbp_season,  # Use season with PBP data
                    week=int(week),
                    pbp_df=pbp_df,  # Pass pre-fetched PBP data
                    schedules_df=schedules_df,  # Pass current season's schedule
                )
                # Extract the features dict from SpreadPredictionFeatures object
                if game_features and hasattr(game_features, 'features'):
                    features[game_id] = game_features.features
                    logger.debug(f"Built features for {game_id}: {len(game_features.features)} features")
                else:
                    feature_errors.append(f"{game_id}: build_spread_features returned None or no features")
            except Exception as e:
                import traceback
                error_msg = f"{game_id}: {type(e).__name__}: {str(e)}"
                feature_errors.append(error_msg)
                logger.error(f"Feature build failed: {error_msg}")
                logger.debug(traceback.format_exc())

        logger.info(f"Step 4: Built features for {len(features)}/{len(games)} games")
        if feature_errors and len(feature_errors) <= 5:
            logger.warning(f"Feature errors: {feature_errors}")
        elif feature_errors:
            logger.warning(f"Feature errors (first 5): {feature_errors[:5]}")

        if not features:
            logger.warning("POLL FAILED: No features built for any game")
            return []

        # 4. Run value detection
        logger.info(f"Step 5: Running value detection on {len(odds_data)} odds entries")
        result = value_detector.scan_spreads(
            games=games,
            odds_data=odds_data,
            features=features,
        )

        value_bets = result.value_bets
        logger.info(f"Step 6: Value detection found {len(value_bets)} value bets (scanned {result.games_scanned} games)")

        # Log threshold info if no bets found
        if not value_bets:
            logger.info(f"No value bets met thresholds: min_edge={value_detector.min_edge}, min_ev={value_detector.min_ev}")
            logger.info("This is normal if market is efficient or model predictions are close to market odds")

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
        logger.info(f"=== ODDS POLL COMPLETE in {elapsed:.1f}s === Found {len(value_bets)} value bets")

        return value_bets

    except Exception as e:
        import traceback
        logger.error(f"POLL FAILED with exception: {e}")
        logger.error(traceback.format_exc())
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


def parse_props(props_data: dict, game: dict) -> list[dict]:
    """
    Parse Odds API player props response into list format.

    Args:
        props_data: Raw response from The Odds API player props endpoint
        game: Game dict with game_id, season, week, home_team, away_team

    Returns:
        List of prop dicts with player_id, player_name, prop_type, line, odds
    """
    props_list = []
    market_map = {
        "player_pass_yds": "passing_yards",
        "player_rush_yds": "rushing_yards",
        "player_reception_yds": "receiving_yards",
        "player_receptions": "receptions",
    }

    for bookmaker in props_data.get("bookmakers", []):
        for market in bookmaker.get("markets", []):
            prop_type = market_map.get(market.get("key"))
            if not prop_type:
                continue

            # Group outcomes by player to get both over/under
            player_outcomes = {}
            for outcome in market.get("outcomes", []):
                player_name = outcome.get("description")
                if not player_name:
                    continue
                if player_name not in player_outcomes:
                    player_outcomes[player_name] = {"line": outcome.get("point")}
                if outcome.get("name") == "Over":
                    player_outcomes[player_name]["over_odds"] = outcome.get("price")
                else:
                    player_outcomes[player_name]["under_odds"] = outcome.get("price")

            for player_name, odds in player_outcomes.items():
                props_list.append({
                    "player_id": player_name,  # Use name as ID for now
                    "player_name": player_name,
                    "prop_type": prop_type,
                    "line": odds.get("line"),
                    "over_odds": odds.get("over_odds"),
                    "under_odds": odds.get("under_odds"),
                    "bookmaker": bookmaker.get("key"),
                    "game_id": game.get("game_id"),
                    "season": game.get("season"),
                    "week": game.get("week"),
                    "team": game.get("home_team"),  # Approximate
                    "opponent": game.get("away_team"),
                })

    return props_list


async def poll_props(
    pipeline: Any,
    feature_pipeline: Any,
    value_detector: Any,
    bankroll_manager: Any,
) -> list:
    """
    Poll player props and detect value bets.

    Runs hourly (separate from spreads) to manage API costs.
    Props use ~5 credits per game vs 1 for spreads.

    Args:
        pipeline: DataPipeline instance
        feature_pipeline: FeaturePipeline instance
        value_detector: ValueDetector instance
        bankroll_manager: BankrollManager instance

    Returns:
        List of ValueBet opportunities found for player props
    """
    from nfl_bets.betting.kelly_calculator import KellyCalculator

    logger.info("=== STARTING PROPS POLL ===")
    start_time = datetime.now()

    try:
        # 1. Get upcoming games from nflverse
        games_df = await pipeline.get_upcoming_games()
        if games_df is None or len(games_df) == 0:
            logger.warning("PROPS POLL: No upcoming games found")
            return []

        games = games_df.to_dicts()
        logger.info(f"Step 1: Found {len(games)} upcoming games for props")

        # 2. Get Odds API events to map game_id -> event_id
        # The Odds API uses its own event IDs (UUIDs), not nflverse game_ids
        try:
            odds_events = await pipeline.odds_api.get_events()
            logger.info(f"Step 2: Got {len(odds_events)} events from Odds API")
        except Exception as e:
            logger.error(f"PROPS POLL: Failed to get Odds API events: {e}")
            return []

        # Build mapping: (home_team, away_team) -> event_id
        event_map = {}
        for event in odds_events:
            home = TEAM_NAME_MAP.get(event.get("home_team", ""), event.get("home_team", "").upper())
            away = TEAM_NAME_MAP.get(event.get("away_team", ""), event.get("away_team", "").upper())
            event_map[(home, away)] = event.get("id")

        # 3. Fetch player props for each game using Odds API event_id
        all_props = []
        for game in games:
            game_id = game.get("game_id")
            home_team = game.get("home_team", "").upper()
            away_team = game.get("away_team", "").upper()

            # Look up the Odds API event_id
            event_id = event_map.get((home_team, away_team))
            if not event_id:
                logger.debug(f"No Odds API event found for {away_team} @ {home_team}")
                continue

            try:
                # Use event_id (not game_id) to fetch props from Odds API
                # Filter to DraftKings only to save API credits
                props_data = await pipeline.odds_api.get_player_props(
                    event_id=event_id,
                    bookmakers=["draftkings"],
                )
                if props_data and props_data.get("bookmakers"):
                    parsed = parse_props(props_data, game)
                    all_props.extend(parsed)
                    logger.debug(f"Got {len(parsed)} props for {away_team} @ {home_team}")
            except Exception as e:
                logger.debug(f"No props for {game_id} (event {event_id}): {e}")

        if not all_props:
            logger.info("PROPS POLL: No player props available from DraftKings")
            return []

        logger.info(f"Step 3: Found {len(all_props)} player props from DraftKings")

        # 4. Determine PBP season from game data
        sample_season = games[0].get("season") if games else None
        pbp_season = int(sample_season) if sample_season else 2025
        logger.info(f"Step 4: Using PBP season {pbp_season} for features")

        # Pre-fetch PBP data once for all players
        pbp_df = await pipeline.get_historical_pbp([pbp_season])
        if pbp_df is None or len(pbp_df) == 0:
            logger.warning(f"PROPS POLL: No PBP data for season {pbp_season}")
            return []

        # 5. Build features for each unique player
        # First, resolve player names to nflverse player_ids
        player_features = {}
        player_id_cache = {}  # Cache player_name -> player_id lookups

        for prop in all_props:
            player_name = prop.get("player_name")
            if not player_name:
                continue

            # Use cache or look up player_id
            if player_name in player_id_cache:
                player_id = player_id_cache[player_name]
            else:
                # Look up real nflverse player_id from name
                position = _get_position_for_prop_type(prop.get("prop_type"))
                player_id = await feature_pipeline.lookup_player_id(
                    player_name=player_name,
                    season=pbp_season,
                    position=position,
                )
                player_id_cache[player_name] = player_id

            if not player_id:
                logger.debug(f"Could not find player_id for {player_name}")
                continue

            # Update the prop with the real player_id
            prop["player_id"] = player_id

            # Skip if we already have features for this player
            if player_id in player_features:
                continue

            week = prop.get("week")
            if not week:
                continue

            try:
                pf = await feature_pipeline.build_prop_features(
                    game_id=prop["game_id"],
                    player_id=player_id,
                    player_name=player_name,
                    prop_type=prop["prop_type"],
                    season=pbp_season,  # Use mapped season
                    week=int(week),
                    opponent_team=prop["opponent"],
                    pbp_df=pbp_df,  # Pass pre-fetched PBP data
                )
                player_features[player_id] = pf.features
            except Exception as e:
                logger.debug(f"Could not build prop features for {player_name}: {e}")

        logger.info(f"Step 5: Built features for {len(player_features)} players")

        # 6. Run prop value detection
        if not player_features:
            logger.warning("PROPS POLL: No player features built, skipping detection")
            return []

        result = value_detector.scan_props(
            props=all_props,
            player_features=player_features,
        )

        value_bets = result.value_bets
        logger.info(f"Step 6: Value detection found {len(value_bets)} prop value bets")

        # 7. Calculate recommended stakes
        kelly = KellyCalculator()
        for bet in value_bets:
            stake = kelly.calculate_stake(
                bankroll=bankroll_manager.current_bankroll,
                win_probability=bet.model_probability,
                odds=bet.odds,
            )
            bet.recommended_stake = stake.recommended_stake

        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"=== PROPS POLL COMPLETE in {elapsed:.1f}s === Found {len(value_bets)} prop bets")

        return value_bets

    except Exception as e:
        import traceback
        logger.error(f"PROPS POLL FAILED: {e}")
        logger.error(traceback.format_exc())
        return []


def _get_position_for_prop_type(prop_type: str) -> str:
    """Get the position hint for player ID lookup based on prop type."""
    position_map = {
        "passing_yards": "QB",
        "rushing_yards": "RB",
        "receiving_yards": "WR",
        "receptions": "WR",
    }
    return position_map.get(prop_type, None)
