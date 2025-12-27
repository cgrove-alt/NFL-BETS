"""Value bets endpoints."""

from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel

router = APIRouter()


def get_val(obj: Any, key: str, default: Any = None) -> Any:
    """
    Universal accessor for both Class Objects and Dictionaries.

    Handles the data type mismatch between:
    - Live data: Python objects with getattr()
    - Fallback data: Dictionaries with dict.get()
    """
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _get_urgency_str(bet: Any) -> str:
    """Extract urgency as uppercase string from bet object or dict."""
    urgency = get_val(bet, 'urgency', 'MEDIUM')
    if hasattr(urgency, 'value'):
        return urgency.value.upper()
    if isinstance(urgency, str):
        return urgency.upper()
    return str(urgency).upper()


def _get_datetime_iso(bet: Any, key: str) -> str:
    """Extract datetime field as ISO string."""
    val = get_val(bet, key)
    if val is None:
        return datetime.now().isoformat()
    if hasattr(val, 'isoformat'):
        return val.isoformat()
    return str(val)


class ValueBetResponse(BaseModel):
    """Response model for a value bet."""

    bet_type: str
    game_id: str
    description: str
    model_probability: float
    model_prediction: float
    bookmaker: str
    odds: int
    implied_probability: float
    line: float
    edge: float
    expected_value: float
    recommended_stake: Optional[float] = None
    urgency: str
    detected_at: str
    expires_at: Optional[str] = None


class ValueBetsListResponse(BaseModel):
    """Response model for value bets list."""

    count: int
    value_bets: list[ValueBetResponse]
    last_poll: Optional[str] = None


@router.get("/value-bets", response_model=ValueBetsListResponse)
async def get_value_bets(
    request: Request,
    min_edge: float = Query(0.0, description="Minimum edge threshold"),
    bet_type: Optional[str] = Query(None, description="Filter by bet type (spread, passing_yards, etc.)"),
    urgency: Optional[str] = Query(None, description="Filter by urgency (LOW, MEDIUM, HIGH, CRITICAL)"),
    limit: int = Query(50, description="Maximum number of bets to return"),
) -> dict[str, Any]:
    """
    Get current value bet opportunities.

    Returns all value bets detected in the last polling cycle,
    optionally filtered by edge, bet type, or urgency.
    """
    app_state = request.app.state.app_state

    # Get value bets from scheduler
    value_bets = app_state.last_value_bets

    # Apply filters using get_val() for both objects and dicts
    filtered_bets = []
    for bet in value_bets:
        # Filter by edge
        bet_edge = get_val(bet, 'edge', 0)
        if bet_edge < min_edge:
            continue

        # Filter by bet type
        if bet_type and get_val(bet, 'bet_type', '') != bet_type:
            continue

        # Filter by urgency
        if urgency:
            bet_urgency = _get_urgency_str(bet)
            if bet_urgency != urgency.upper():
                continue

        filtered_bets.append(bet)

    # Limit results
    filtered_bets = filtered_bets[:limit]

    # Convert to response format using get_val() for universal access
    response_bets = []
    for bet in filtered_bets:
        response_bets.append(
            ValueBetResponse(
                bet_type=get_val(bet, 'bet_type', 'unknown'),
                game_id=get_val(bet, 'game_id', ''),
                description=get_val(bet, 'description', ''),
                model_probability=get_val(bet, 'model_probability', 0.0),
                model_prediction=get_val(bet, 'model_prediction', 0.0),
                bookmaker=get_val(bet, 'bookmaker', 'unknown'),
                odds=get_val(bet, 'odds', 0),
                implied_probability=get_val(bet, 'implied_probability', 0.0),
                line=get_val(bet, 'line', 0.0),
                edge=get_val(bet, 'edge', 0.0),
                expected_value=get_val(bet, 'expected_value', 0.0),
                recommended_stake=get_val(bet, 'recommended_stake'),
                urgency=_get_urgency_str(bet),
                detected_at=_get_datetime_iso(bet, 'detected_at'),
                expires_at=None,  # ValueBet doesn't have expires_at field
            )
        )

    # Get last poll time from scheduler
    last_poll = None
    if app_state.scheduler:
        job_status = app_state.scheduler.get_job_status()
        poll_status = job_status.get("poll_odds", {})
        if poll_status.get("last_run"):
            last_poll = poll_status["last_run"].isoformat()

    return {
        "count": len(response_bets),
        "value_bets": response_bets,
        "last_poll": last_poll,
    }


@router.get("/value-bets/debug")
async def debug_value_detection(request: Request) -> dict[str, Any]:
    """
    Debug endpoint to test the value detection pipeline.

    Returns detailed diagnostic info about each step of the pipeline.
    """
    import logging
    from nfl_bets.scheduler.jobs import _transform_odds_data

    app_state = request.app.state.app_state
    logger = logging.getLogger(__name__)

    debug_info = {
        "code_version": "v8-edge-calc-test",  # Added edge_test to debug endpoint
        "pipeline_initialized": app_state.pipeline is not None,
        "feature_pipeline_initialized": app_state.feature_pipeline is not None,
        "value_detector_initialized": app_state.value_detector is not None,
        "scheduler_running": app_state.scheduler.is_running if app_state.scheduler else False,
        "spread_model_loaded": False,
        "prop_models_loaded": [],
        "games_count": 0,
        "sample_games": [],
        "raw_odds_count": 0,
        "sample_raw_odds": [],
        "transformed_odds_count": 0,
        "sample_transformed_odds": [],
        "team_name_mismatches": [],
        "features_built": 0,
        "feature_errors": [],
        "value_bets_in_memory": 0,
        "min_edge_threshold": None,
        "min_ev_threshold": None,
        "errors": [],
    }

    try:
        # Check spread model
        if app_state.value_detector:
            if app_state.value_detector.spread_model:
                debug_info["spread_model_loaded"] = True
            debug_info["prop_models_loaded"] = list(app_state.value_detector.prop_models.keys())
            debug_info["min_edge_threshold"] = app_state.value_detector.min_edge
            debug_info["min_ev_threshold"] = app_state.value_detector.min_ev

        # Get games
        if app_state.pipeline:
            games_df = await app_state.pipeline.get_upcoming_games()
            if games_df is not None:
                games = games_df.to_dicts()
                debug_info["games_count"] = len(games)
                debug_info["sample_games"] = [
                    {
                        "game_id": g.get("game_id"),
                        "home_team": g.get("home_team"),
                        "away_team": g.get("away_team"),
                        "season": g.get("season"),
                        "week": g.get("week"),
                    }
                    for g in games[:3]
                ]

                # Get raw odds
                raw_odds = await app_state.pipeline.get_game_odds()
                debug_info["raw_odds_count"] = len(raw_odds) if raw_odds else 0

                if raw_odds:
                    debug_info["sample_raw_odds"] = [
                        {
                            "home_team": o.get("home_team"),
                            "away_team": o.get("away_team"),
                            "bookmakers_count": len(o.get("bookmakers", [])),
                        }
                        for o in raw_odds[:3]
                    ]

                    # Transform odds and track mismatches
                    transformed = _transform_odds_data(raw_odds, games)
                    debug_info["transformed_odds_count"] = len(transformed)

                    if transformed:
                        debug_info["sample_transformed_odds"] = transformed[:3]

                    # Identify team name mismatches
                    game_teams = set()
                    for g in games:
                        game_teams.add(g.get("home_team", "").upper())
                        game_teams.add(g.get("away_team", "").upper())

                    for odds_event in raw_odds:
                        home = odds_event.get("home_team", "")
                        away = odds_event.get("away_team", "")
                        # These would be the normalized versions
                        from nfl_bets.scheduler.jobs import TEAM_NAME_MAP
                        home_norm = TEAM_NAME_MAP.get(home, home.upper())
                        away_norm = TEAM_NAME_MAP.get(away, away.upper())

                        if home_norm not in game_teams or away_norm not in game_teams:
                            debug_info["team_name_mismatches"].append({
                                "odds_home": home,
                                "odds_away": away,
                                "normalized_home": home_norm,
                                "normalized_away": away_norm,
                                "home_in_games": home_norm in game_teams,
                                "away_in_games": away_norm in game_teams,
                            })

                    # Limit mismatches shown
                    debug_info["team_name_mismatches"] = debug_info["team_name_mismatches"][:5]

                # TEST: Try to build features for one game to see if it works
                if games and app_state.feature_pipeline:
                    test_game = games[0]
                    test_game_id = test_game.get("game_id")
                    test_season = test_game.get("season", 2025)
                    test_week = test_game.get("week", 17)

                    debug_info["feature_test"] = {
                        "game_id": test_game_id,
                        "season": test_season,
                        "week": test_week,
                        "status": "pending",
                        "error": None,
                    }

                    try:
                        # Try to fetch PBP data
                        import polars as pl
                        pbp_df = None
                        for season_try in [test_season, test_season - 1, 2024]:
                            pbp_df = await app_state.pipeline.get_historical_pbp([season_try])
                            if pbp_df is not None and len(pbp_df) > 0:
                                debug_info["feature_test"]["pbp_season"] = season_try
                                debug_info["feature_test"]["pbp_rows"] = len(pbp_df)
                                break

                        if pbp_df is None or len(pbp_df) == 0:
                            debug_info["feature_test"]["status"] = "failed"
                            debug_info["feature_test"]["error"] = "No PBP data available for any season"
                        else:
                            # Try to fetch schedules
                            schedules_df = await app_state.pipeline.get_schedules([debug_info["feature_test"]["pbp_season"]])
                            if schedules_df is None:
                                schedules_df = pl.DataFrame(games)
                                debug_info["feature_test"]["schedules_source"] = "games_fallback"
                            else:
                                debug_info["feature_test"]["schedules_source"] = "nflverse"
                                debug_info["feature_test"]["schedules_rows"] = len(schedules_df)

                            # Try to build features
                            test_features = await app_state.feature_pipeline.build_spread_features(
                                game_id=test_game_id,
                                home_team=test_game.get("home_team"),
                                away_team=test_game.get("away_team"),
                                season=debug_info["feature_test"]["pbp_season"],
                                week=int(test_week),
                                pbp_df=pbp_df,
                                schedules_df=schedules_df,
                            )

                            if test_features and hasattr(test_features, 'features'):
                                debug_info["feature_test"]["status"] = "success"
                                debug_info["feature_test"]["feature_count"] = len(test_features.features)
                                debug_info["features_built"] = 1  # At least one works

                                # EDGE CALCULATION TEST: Run prediction with line to see actual values
                                if app_state.value_detector and app_state.value_detector.spread_model:
                                    # Find odds for this game
                                    test_odds = None
                                    for o in transformed:
                                        if o.get("game_id") == test_game_id:
                                            test_odds = o
                                            break

                                    if test_odds:
                                        line = test_odds.get("home_spread")
                                        home_odds = test_odds.get("home_odds")
                                        away_odds = test_odds.get("away_odds")

                                        # Run prediction WITH line
                                        test_pred = app_state.value_detector.spread_model.predict_game(
                                            features=test_features.features,
                                            game_id=test_game_id,
                                            home_team=test_game.get("home_team"),
                                            away_team=test_game.get("away_team"),
                                            line=line,
                                        )

                                        # Calculate edge manually
                                        from nfl_bets.betting.odds_converter import american_to_implied_probability, remove_vig
                                        fair_prob_home, fair_prob_away = remove_vig(home_odds, away_odds)
                                        edge_home = test_pred.home_cover_prob - float(fair_prob_home)
                                        edge_away = test_pred.away_cover_prob - float(fair_prob_away)

                                        debug_info["edge_test"] = {
                                            "game_id": test_game_id,
                                            "line": line,
                                            "home_odds": home_odds,
                                            "away_odds": away_odds,
                                            "predicted_spread": round(test_pred.predicted_spread, 2),
                                            "home_cover_prob": round(test_pred.home_cover_prob, 4),
                                            "away_cover_prob": round(test_pred.away_cover_prob, 4),
                                            "fair_prob_home": round(float(fair_prob_home), 4),
                                            "fair_prob_away": round(float(fair_prob_away), 4),
                                            "edge_home": round(edge_home, 4),
                                            "edge_away": round(edge_away, 4),
                                            "threshold": app_state.value_detector.min_edge,
                                            "home_passes": edge_home >= app_state.value_detector.min_edge,
                                            "away_passes": edge_away >= app_state.value_detector.min_edge,
                                        }
                            else:
                                debug_info["feature_test"]["status"] = "failed"
                                debug_info["feature_test"]["error"] = "build_spread_features returned None"
                    except Exception as e:
                        import traceback
                        debug_info["feature_test"]["status"] = "failed"
                        debug_info["feature_test"]["error"] = f"{type(e).__name__}: {str(e)}"
                        debug_info["feature_test"]["traceback"] = traceback.format_exc()

        # Get current value bets in memory
        debug_info["value_bets_in_memory"] = len(app_state.last_value_bets)

        # Show sample value bets if any (using get_val for universal access)
        if app_state.last_value_bets:
            debug_info["sample_value_bets"] = [
                {
                    "game_id": get_val(bet, 'game_id', ''),  # CRITICAL: Show game_id for matching debug
                    "description": get_val(bet, 'description', ''),
                    "edge": get_val(bet, 'edge', 0),
                    "model_probability": get_val(bet, 'model_probability', 0),
                    "odds": get_val(bet, 'odds', 0),
                }
                for bet in app_state.last_value_bets[:3]
            ]

        # Check scheduler job status
        if app_state.scheduler:
            try:
                job_status = app_state.scheduler.get_job_status()
                poll_status = job_status.get("poll_odds", {})
                debug_info["last_poll"] = {
                    "status": poll_status.get("last_status"),
                    "time": poll_status.get("last_run").isoformat() if poll_status.get("last_run") else None,
                    "error": poll_status.get("last_error"),
                    "run_count": poll_status.get("run_count", 0),
                }
            except Exception as e:
                debug_info["scheduler_status_error"] = str(e)

    except Exception as e:
        import traceback
        debug_info["errors"].append(f"{type(e).__name__}: {str(e)}")
        debug_info["traceback"] = traceback.format_exc()

    return debug_info


@router.get("/value-bets/summary")
async def get_value_bets_summary(request: Request) -> dict[str, Any]:
    """
    Get summary statistics for current value bets.

    Returns counts by urgency, bet type, and average edge.
    """
    app_state = request.app.state.app_state
    value_bets = app_state.last_value_bets

    if not value_bets:
        return {
            "total_bets": 0,
            "by_urgency": {},
            "by_bet_type": {},
            "average_edge": 0,
            "total_expected_value": 0,
        }

    # Count by urgency (using get_val for universal access)
    by_urgency = {}
    for bet in value_bets:
        urgency = _get_urgency_str(bet)
        by_urgency[urgency] = by_urgency.get(urgency, 0) + 1

    # Count by bet type
    by_bet_type = {}
    for bet in value_bets:
        bet_type = get_val(bet, 'bet_type', 'unknown')
        by_bet_type[bet_type] = by_bet_type.get(bet_type, 0) + 1

    # Calculate averages
    total_edge = sum(get_val(bet, 'edge', 0) for bet in value_bets)
    total_ev = sum(get_val(bet, 'expected_value', 0) for bet in value_bets)

    return {
        "total_bets": len(value_bets),
        "by_urgency": by_urgency,
        "by_bet_type": by_bet_type,
        "average_edge": total_edge / len(value_bets) if value_bets else 0,
        "total_expected_value": total_ev,
    }
