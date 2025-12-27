"""Games endpoints - upcoming games with value bet aggregation."""

import logging
from collections import defaultdict
from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel

logger = logging.getLogger(__name__)

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
    # Cold start / demo mode flags
    is_demo: bool = False
    is_initializing: bool = False
    retry_after_seconds: Optional[int] = None
    # Fallback mode flag - indicates emergency hardcoded data
    is_fallback: bool = False


def _format_bet(bet, game_id_override: str = None) -> dict:
    """
    Format a bet object into a dictionary matching the ValueBet frontend interface.

    Args:
        bet: The bet object (can be ValueBet, FallbackValueBet, or similar)
        game_id_override: Optional game_id to use instead of bet's game_id

    Returns:
        Dictionary with all ValueBet fields
    """
    # Get urgency value - handle enum or string
    urgency_val = getattr(bet, "urgency", "medium")
    if hasattr(urgency_val, "value"):
        urgency_str = urgency_val.value
    else:
        urgency_str = str(urgency_val)

    # Get detected_at - handle datetime or None
    detected_at = getattr(bet, "detected_at", None)
    if detected_at is not None:
        if isinstance(detected_at, datetime):
            detected_at_str = detected_at.isoformat()
        else:
            detected_at_str = str(detected_at)
    else:
        detected_at_str = datetime.now().isoformat()

    # Get expires_at - handle datetime or None
    expires_at = getattr(bet, "expires_at", None)
    if expires_at is not None:
        if isinstance(expires_at, datetime):
            expires_at_str = expires_at.isoformat()
        else:
            expires_at_str = str(expires_at)
    else:
        expires_at_str = None

    return {
        "game_id": game_id_override or getattr(bet, "game_id", ""),
        "bet_type": getattr(bet, "bet_type", ""),
        "description": getattr(bet, "description", ""),
        "model_probability": getattr(bet, "model_probability", 0),
        "model_prediction": getattr(bet, "model_prediction", 0),
        "bookmaker": getattr(bet, "bookmaker", ""),
        "odds": getattr(bet, "odds", 0),
        "implied_probability": getattr(bet, "implied_probability", 0),
        "line": getattr(bet, "line", 0),
        "edge": getattr(bet, "edge", 0),
        "expected_value": getattr(bet, "expected_value", 0),
        "recommended_stake": getattr(bet, "recommended_stake", None),
        "urgency": urgency_str,
        "detected_at": detected_at_str,
        "expires_at": expires_at_str,
    }


def _build_demo_games(app_state) -> list[GameResponse]:
    """
    Build game responses from demo/mock data.

    Args:
        app_state: Application state with mock games

    Returns:
        List of GameResponse objects from mock data
    """
    response_games = []

    # Get value bets and group by game
    value_bets = app_state.last_value_bets
    bets_by_game = defaultdict(list)
    for bet in value_bets:
        game_id = getattr(bet, "game_id", "")
        bets_by_game[game_id].append(bet)

    for game in app_state._mock_games:
        game_id = game.get("game_id", "")
        game_bets = bets_by_game.get(game_id, [])

        # Find best edge bet
        best_bet = None
        if game_bets:
            best_bet = max(game_bets, key=lambda b: getattr(b, "edge", 0))

        response_games.append(
            GameResponse(
                game_id=game_id,
                home_team=game.get("home_team", ""),
                away_team=game.get("away_team", ""),
                kickoff=game.get("kickoff", ""),
                week=game.get("week", 1),
                season=game.get("season", 2024),
                value_bet_count=len(game_bets),
                best_edge=getattr(best_bet, "edge", None) if best_bet else None,
                best_bet_description=getattr(best_bet, "description", None) if best_bet else None,
                model_prediction=getattr(best_bet, "model_prediction", None) if best_bet else None,
                model_confidence=getattr(best_bet, "model_probability", None) if best_bet else None,
                vegas_line=game.get("spread"),
            )
        )

    # Sort by kickoff time, then by value bet count (most bets first)
    response_games.sort(key=lambda g: (g.kickoff, -g.value_bet_count))

    return response_games


def _build_fallback_games(app_state) -> list[GameResponse]:
    """
    Build game responses from HARD FALLBACK data.

    Called when:
    - Scheduler is dead
    - Live data fetch failed
    - No games available from pipeline

    Args:
        app_state: Application state with fallback methods

    Returns:
        List of GameResponse objects from hardcoded fallback data
    """
    logger.warning("🚨 FALLBACK: Building games from hardcoded fallback data")

    response_games = []

    # Get fallback games and bets
    fallback_games = app_state.get_fallback_games()
    fallback_bets = app_state.get_fallback_data()

    # Group bets by game
    bets_by_game = defaultdict(list)
    for bet in fallback_bets:
        game_id = getattr(bet, "game_id", "")
        bets_by_game[game_id].append(bet)

    for game in fallback_games:
        game_id = game.get("game_id", "")
        game_bets = bets_by_game.get(game_id, [])

        # Find best edge bet
        best_bet = None
        if game_bets:
            best_bet = max(game_bets, key=lambda b: getattr(b, "edge", 0))

        response_games.append(
            GameResponse(
                game_id=game_id,
                home_team=game.get("home_team", ""),
                away_team=game.get("away_team", ""),
                kickoff=game.get("kickoff", ""),
                week=game.get("week", 17),
                season=game.get("season", 2024),
                value_bet_count=len(game_bets),
                best_edge=getattr(best_bet, "edge", None) if best_bet else None,
                best_bet_description=getattr(best_bet, "description", None) if best_bet else None,
                model_prediction=getattr(best_bet, "model_prediction", None) if best_bet else None,
                model_confidence=getattr(best_bet, "model_probability", None) if best_bet else None,
                vegas_line=game.get("spread"),
            )
        )

    # Sort by kickoff time, then by value bet count (most bets first)
    response_games.sort(key=lambda g: (g.kickoff, -g.value_bet_count))

    return response_games


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

    Also returns status flags:
    - is_demo: True if serving mock/demo data
    - is_initializing: True if backend is still warming up
    - retry_after_seconds: Suggested retry delay if initializing
    """
    app_state = request.app.state.app_state

    # Get status flags
    is_demo = getattr(app_state, "_demo_mode", False)
    is_initializing = getattr(app_state, "_is_initializing", True)

    # DEMO MODE: Return mock games immediately
    if is_demo and app_state._mock_games:
        logger.info("🎭 Returning DEMO MODE games")
        demo_games = _build_demo_games(app_state)

        # Filter by week if specified
        if week is not None:
            demo_games = [g for g in demo_games if g.week == week]

        return {
            "count": len(demo_games),
            "games": demo_games,
            "is_demo": True,
            "is_initializing": False,
            "retry_after_seconds": None,
        }

    # COLD START: If still initializing and no data, tell frontend to retry
    if is_initializing and not app_state.last_value_bets:
        # Check if we have cached data to show
        cached_bets = getattr(app_state, "_cached_bets_raw", None)
        if not cached_bets:
            logger.info("⏳ System initializing - advising frontend to retry in 5s")
            return {
                "count": 0,
                "games": [],
                "is_demo": False,
                "is_initializing": True,
                "retry_after_seconds": 5,
            }

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
            logger.warning(f"Failed to get enriched games: {e}")

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

    # HARD FALLBACK: If response_games is empty, use fallback data
    is_fallback = False
    if len(response_games) == 0:
        logger.warning("🚨 No games from pipeline - activating HARD FALLBACK")
        response_games = _build_fallback_games(app_state)
        is_fallback = True

    return {
        "count": len(response_games),
        "games": response_games,
        "is_demo": False,
        "is_initializing": is_initializing,
        "retry_after_seconds": 5 if is_initializing and len(response_games) == 0 else None,
        "is_fallback": is_fallback,
    }


@router.get("/games/{game_id}")
async def get_game_detail(
    request: Request,
    game_id: str,
) -> dict[str, Any]:
    """
    Get detailed information for a specific game.

    Returns the game with all associated value bets.

    ROBUST FALLBACK LOGIC:
    1. Check if we're in demo mode -> use mock data
    2. Check if we're in fallback mode -> use fallback data directly (skip pipeline)
    3. Try pipeline -> if fails, fall back to synthetic recovery
    """
    app_state = request.app.state.app_state

    # Check demo mode first
    is_demo = getattr(app_state, "_demo_mode", False)

    if is_demo and app_state._mock_games:
        # Find game in mock data
        for game in app_state._mock_games:
            if game.get("game_id") == game_id:
                # Get bets for this game
                game_bets = [
                    bet for bet in app_state.last_value_bets
                    if getattr(bet, "game_id", "") == game_id
                ]

                # Format bets with game_id included
                formatted_bets = [_format_bet(bet, game_id) for bet in game_bets]
                formatted_bets.sort(key=lambda b: b["edge"], reverse=True)

                return {
                    "game_id": game_id,
                    "home_team": game.get("home_team", ""),
                    "away_team": game.get("away_team", ""),
                    "kickoff": game.get("kickoff", ""),
                    "week": game.get("week", 1),
                    "season": game.get("season", 2024),
                    "value_bets": formatted_bets,
                    "value_bet_count": len(formatted_bets),
                    "is_demo": True,
                    "is_fallback": False,
                }

        return {"error": "Game not found", "game_id": game_id, "is_demo": True}

    # =========================================================================
    # ROBUST FALLBACK: Check if we're in fallback mode BEFORE querying pipeline
    # This prevents 404 errors when the system is not fully initialized
    # =========================================================================
    is_initialized = getattr(app_state, "_initialized", False)
    using_fallback = getattr(app_state, "_using_fallback", False)

    # Force fallback mode check by accessing last_value_bets (triggers _using_fallback flag)
    _ = app_state.last_value_bets
    using_fallback = getattr(app_state, "_using_fallback", False)

    if not is_initialized or using_fallback:
        logger.info(f"🔄 FALLBACK MODE: Serving game {game_id} from fallback data (initialized={is_initialized}, using_fallback={using_fallback})")

        # Get fallback games and bets directly - DO NOT query pipeline
        fallback_games = app_state.get_fallback_games()
        fallback_bets = app_state.get_fallback_data()

        # Find the specific game
        target_game = None
        for fg in fallback_games:
            if fg.get("game_id") == game_id:
                target_game = fg
                break

        if target_game:
            # Filter bets for this game
            game_bets = [bet for bet in fallback_bets if getattr(bet, "game_id", "") == game_id]

            # Format bets with game_id included
            formatted_bets = [_format_bet(bet, game_id) for bet in game_bets]
            formatted_bets.sort(key=lambda b: b["edge"], reverse=True)

            return {
                "game_id": game_id,
                "home_team": target_game.get("home_team", ""),
                "away_team": target_game.get("away_team", ""),
                "kickoff": target_game.get("kickoff", ""),
                "week": target_game.get("week", 17),
                "season": target_game.get("season", 2024),
                "value_bets": formatted_bets,
                "value_bet_count": len(formatted_bets),
                "is_demo": False,
                "is_fallback": True,
            }
        else:
            # Game not in fallback data - try synthetic reconstruction
            logger.warning(f"⚠️ Game {game_id} not in fallback games, attempting synthetic recovery")

    # =========================================================================
    # NORMAL MODE: Try to get game from pipeline
    # =========================================================================
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
        except Exception as e:
            logger.warning(f"Pipeline query failed for game {game_id}: {e}")

    # Get value bets for this game
    value_bets = app_state.last_value_bets
    game_bets = [bet for bet in value_bets if getattr(bet, "game_id", "") == game_id]

    # =========================================================================
    # FALLBACK RECOVERY: If pipeline failed, reconstruct game from bets or fallback data
    # =========================================================================
    is_fallback = False
    if not game_data:
        logger.warning(f"🔄 Pipeline returned no data for game {game_id} - attempting fallback recovery")

        # First, try to find game in fallback games
        fallback_games = app_state.get_fallback_games()
        for fg in fallback_games:
            if fg.get("game_id") == game_id:
                logger.info(f"✅ Found game {game_id} in fallback games")
                # Create synthetic game object from fallback data
                game_data = type("SyntheticGame", (), {
                    "game_id": game_id,
                    "home_team": fg.get("home_team", "Unknown"),
                    "away_team": fg.get("away_team", "Unknown"),
                    "kickoff": fg.get("kickoff", datetime.now().isoformat()),
                    "week": fg.get("week", 17),
                    "season": fg.get("season", 2024),
                })()
                is_fallback = True
                break

        # If still no game_data, try to reconstruct from bet metadata
        if not game_data and game_bets:
            logger.info(f"✅ Reconstructing game {game_id} from bet metadata")

            # Parse game_id format: "YYYY_WW_AWAY_HOME" (e.g., "2024_17_BAL_KC")
            parts = game_id.split("_")
            if len(parts) >= 4:
                season = int(parts[0]) if parts[0].isdigit() else 2024
                week = int(parts[1]) if parts[1].isdigit() else 17
                away_team = parts[2]
                home_team = parts[3]
            else:
                # Fallback parsing
                season = 2024
                week = 17
                home_team = "Unknown"
                away_team = "Unknown"

            # Try to extract kickoff from bet if available
            sample_bet = game_bets[0]
            kickoff = getattr(sample_bet, "detected_at", datetime.now())
            if isinstance(kickoff, datetime):
                kickoff = kickoff.isoformat()

            game_data = type("SyntheticGame", (), {
                "game_id": game_id,
                "home_team": home_team,
                "away_team": away_team,
                "kickoff": kickoff,
                "week": week,
                "season": season,
            })()
            is_fallback = True

        # If STILL no game_data, return error
        if not game_data:
            logger.error(f"❌ Could not recover game {game_id} - no fallback or bets available")
            return {"error": "Game not found", "game_id": game_id, "is_fallback": False}

    # =========================================================================
    # FORMAT RESPONSE
    # =========================================================================

    # Format bets with game_id included
    formatted_bets = [_format_bet(bet, game_id) for bet in game_bets]
    formatted_bets.sort(key=lambda b: b["edge"], reverse=True)

    # Format kickoff
    kickoff_val = getattr(game_data, "kickoff", datetime.now())
    kickoff_str = kickoff_val.isoformat() if isinstance(kickoff_val, datetime) else str(kickoff_val)

    return {
        "game_id": game_id,
        "home_team": getattr(game_data, "home_team", "Unknown"),
        "away_team": getattr(game_data, "away_team", "Unknown"),
        "kickoff": kickoff_str,
        "week": getattr(game_data, "week", 17),
        "season": getattr(game_data, "season", 2024),
        "value_bets": formatted_bets,
        "value_bet_count": len(formatted_bets),
        "is_demo": False,
        "is_fallback": is_fallback,
    }
