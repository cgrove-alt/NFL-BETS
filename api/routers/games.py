"""Games endpoints - upcoming games with value bet aggregation."""

from collections import defaultdict
from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel

router = APIRouter()


class GameResponse(BaseModel):
    """Response model for a game with value bet summary."""

    game_id: str
    home_team: str
    away_team: str
    kickoff: str
    week: int
    season: int
    value_bet_count: int
    best_edge: Optional[float] = None
    best_bet_description: Optional[str] = None
    model_prediction: Optional[float] = None
    model_confidence: Optional[float] = None
    vegas_line: Optional[float] = None


class GamesListResponse(BaseModel):
    """Response model for games list."""

    count: int
    games: list[GameResponse]


@router.get("/games", response_model=GamesListResponse)
async def get_games(
    request: Request,
    week: Optional[int] = Query(None, description="Filter by week number"),
) -> dict[str, Any]:
    """
    Get upcoming games with value bet summaries.

    Returns all upcoming games with aggregated value bet info:
    - Number of value bets per game
    - Best edge available
    - Model prediction vs Vegas line
    """
    app_state = request.app.state.app_state

    # Get upcoming games from pipeline
    games_data = []
    if app_state.pipeline:
        try:
            enriched_games = await app_state.pipeline.get_upcoming_games_enriched(
                include_odds=True,
                include_dvoa=False,  # Skip for performance
                include_health=False,
            )
            games_data = enriched_games
        except Exception as e:
            # Log but continue - we can still show games without enriched data
            import logging
            logging.getLogger(__name__).warning(f"Failed to get enriched games: {e}")

    # Get value bets and group by game
    value_bets = app_state.last_value_bets
    bets_by_game = defaultdict(list)
    for bet in value_bets:
        bets_by_game[bet.game_id].append(bet)

    # Build response
    response_games = []

    for game in games_data:
        game_id = game.game_id
        game_bets = bets_by_game.get(game_id, [])

        # Find best edge bet
        best_bet = None
        if game_bets:
            best_bet = max(game_bets, key=lambda b: b.edge)

        # Extract Vegas line from odds if available
        vegas_line = None
        if game.odds and isinstance(game.odds, dict):
            bookmakers = game.odds.get("bookmakers", [])
            for book in bookmakers:
                markets = book.get("markets", [])
                for market in markets:
                    if market.get("key") == "spreads":
                        outcomes = market.get("outcomes", [])
                        for outcome in outcomes:
                            if outcome.get("name") == game.home_team:
                                vegas_line = outcome.get("point")
                                break
                if vegas_line is not None:
                    break

        # Get model prediction from best bet if available
        model_prediction = None
        model_confidence = None
        if best_bet:
            model_prediction = best_bet.model_prediction
            model_confidence = best_bet.model_probability

        # Filter by week if specified
        if week is not None and game.week != week:
            continue

        # Format kickoff time
        kickoff_str = game.kickoff.isoformat() if isinstance(game.kickoff, datetime) else str(game.kickoff)

        response_games.append(
            GameResponse(
                game_id=game_id,
                home_team=game.home_team,
                away_team=game.away_team,
                kickoff=kickoff_str,
                week=game.week,
                season=game.season,
                value_bet_count=len(game_bets),
                best_edge=best_bet.edge if best_bet else None,
                best_bet_description=best_bet.description if best_bet else None,
                model_prediction=model_prediction,
                model_confidence=model_confidence,
                vegas_line=vegas_line,
            )
        )

    # Sort by kickoff time, then by value bet count (most bets first)
    response_games.sort(key=lambda g: (g.kickoff, -g.value_bet_count))

    return {
        "count": len(response_games),
        "games": response_games,
    }


@router.get("/games/{game_id}")
async def get_game_detail(
    request: Request,
    game_id: str,
) -> dict[str, Any]:
    """
    Get detailed information for a specific game.

    Returns the game with all associated value bets.
    """
    app_state = request.app.state.app_state

    # Find the game
    game_data = None
    if app_state.pipeline:
        try:
            enriched_games = await app_state.pipeline.get_upcoming_games_enriched(
                include_odds=True,
                include_dvoa=False,
                include_health=False,
            )
            for game in enriched_games:
                if game.game_id == game_id:
                    game_data = game
                    break
        except Exception:
            pass

    if not game_data:
        return {"error": "Game not found", "game_id": game_id}

    # Get value bets for this game
    value_bets = app_state.last_value_bets
    game_bets = [bet for bet in value_bets if bet.game_id == game_id]

    # Format bets
    formatted_bets = []
    for bet in game_bets:
        formatted_bets.append({
            "bet_type": bet.bet_type,
            "description": bet.description,
            "model_probability": bet.model_probability,
            "model_prediction": bet.model_prediction,
            "bookmaker": bet.bookmaker,
            "odds": bet.odds,
            "implied_probability": bet.implied_probability,
            "line": bet.line,
            "edge": bet.edge,
            "expected_value": bet.expected_value,
            "recommended_stake": bet.recommended_stake,
            "urgency": bet.urgency.value if hasattr(bet.urgency, "value") else str(bet.urgency),
            "detected_at": bet.detected_at.isoformat() if bet.detected_at else None,
        })

    # Sort by edge descending
    formatted_bets.sort(key=lambda b: b["edge"], reverse=True)

    # Format kickoff
    kickoff_str = game_data.kickoff.isoformat() if isinstance(game_data.kickoff, datetime) else str(game_data.kickoff)

    return {
        "game_id": game_id,
        "home_team": game_data.home_team,
        "away_team": game_data.away_team,
        "kickoff": kickoff_str,
        "week": game_data.week,
        "season": game_data.season,
        "value_bets": formatted_bets,
        "value_bet_count": len(formatted_bets),
    }
