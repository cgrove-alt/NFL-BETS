"""Value bets endpoints."""

from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel

router = APIRouter()


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

    # Apply filters
    filtered_bets = []
    for bet in value_bets:
        # Filter by edge
        if bet.edge < min_edge:
            continue

        # Filter by bet type
        if bet_type and bet.bet_type != bet_type:
            continue

        # Filter by urgency
        if urgency:
            bet_urgency = bet.urgency.value if hasattr(bet.urgency, "value") else str(bet.urgency)
            if bet_urgency.upper() != urgency.upper():
                continue

        filtered_bets.append(bet)

    # Limit results
    filtered_bets = filtered_bets[:limit]

    # Convert to response format
    response_bets = []
    for bet in filtered_bets:
        response_bets.append(
            ValueBetResponse(
                bet_type=bet.bet_type,
                game_id=bet.game_id,
                description=bet.description,
                model_probability=bet.model_probability,
                model_prediction=bet.model_prediction,
                bookmaker=bet.bookmaker,
                odds=bet.odds,
                implied_probability=bet.implied_probability,
                line=bet.line,
                edge=bet.edge,
                expected_value=bet.expected_value,
                recommended_stake=bet.recommended_stake,
                urgency=bet.urgency.value if hasattr(bet.urgency, "value") else str(bet.urgency),
                detected_at=bet.detected_at.isoformat() if bet.detected_at else datetime.now().isoformat(),
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
        "code_version": "v2-season-fix",  # Update this to verify deployment
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

        # Get current value bets in memory
        debug_info["value_bets_in_memory"] = len(app_state.last_value_bets)

        # Check scheduler job status
        if app_state.scheduler:
            try:
                job_status = app_state.scheduler.get_job_status()
                poll_status = job_status.get("poll_odds", {})
                debug_info["last_poll_status"] = poll_status.get("status")
                if poll_status.get("last_run"):
                    debug_info["last_poll_time"] = poll_status["last_run"].isoformat()
                debug_info["poll_error"] = poll_status.get("error")
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

    # Count by urgency
    by_urgency = {}
    for bet in value_bets:
        urgency = bet.urgency.value if hasattr(bet.urgency, "value") else str(bet.urgency)
        by_urgency[urgency] = by_urgency.get(urgency, 0) + 1

    # Count by bet type
    by_bet_type = {}
    for bet in value_bets:
        by_bet_type[bet.bet_type] = by_bet_type.get(bet.bet_type, 0) + 1

    # Calculate averages
    total_edge = sum(bet.edge for bet in value_bets)
    total_ev = sum(bet.expected_value for bet in value_bets)

    return {
        "total_bets": len(value_bets),
        "by_urgency": by_urgency,
        "by_bet_type": by_bet_type,
        "average_edge": total_edge / len(value_bets) if value_bets else 0,
        "total_expected_value": total_ev,
    }
