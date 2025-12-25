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

    Runs a single poll cycle and returns detailed diagnostic info.
    """
    import logging
    from nfl_bets.scheduler.jobs import poll_odds, _transform_odds_data

    app_state = request.app.state.app_state
    logger = logging.getLogger(__name__)

    debug_info = {
        "pipeline_initialized": app_state.pipeline is not None,
        "value_detector_initialized": app_state.value_detector is not None,
        "spread_model_loaded": False,
        "games_count": 0,
        "raw_odds_count": 0,
        "transformed_odds_count": 0,
        "features_built": 0,
        "value_bets_found": 0,
        "min_edge_threshold": None,
        "errors": [],
    }

    try:
        # Check spread model
        if app_state.value_detector and app_state.value_detector.spread_model:
            debug_info["spread_model_loaded"] = True

        # Check min_edge setting
        if app_state.value_detector:
            debug_info["min_edge_threshold"] = app_state.value_detector.min_edge

        # Get games
        if app_state.pipeline:
            games_df = await app_state.pipeline.get_upcoming_games()
            if games_df is not None:
                games = games_df.to_dicts()
                debug_info["games_count"] = len(games)

                # Get raw odds
                raw_odds = await app_state.pipeline.get_game_odds()
                debug_info["raw_odds_count"] = len(raw_odds) if raw_odds else 0

                # Transform odds
                if raw_odds and games:
                    transformed = _transform_odds_data(raw_odds, games)
                    debug_info["transformed_odds_count"] = len(transformed)

                    # Show sample transformed odds
                    if transformed:
                        debug_info["sample_transformed_odds"] = transformed[:3]

        # Get current value bets
        debug_info["value_bets_found"] = len(app_state.last_value_bets)

    except Exception as e:
        debug_info["errors"].append(str(e))

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
