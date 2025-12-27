"""Games endpoints - upcoming games with value bet aggregation."""

import logging
from collections import defaultdict
from datetime import datetime
from typing import Any, Optional, Union

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()


# =============================================================================
# UNIVERSAL DATA ACCESSOR
# =============================================================================
def get_val(obj: Any, key: str, default: Any = None) -> Any:
    """
    Safely retrieves a value from either a Class Object or a Dictionary.

    This is CRITICAL for handling the data type mismatch between:
    - Live Data from pipeline (Python Objects accessed via getattr)
    - Fallback Data (Dictionaries accessed via ['key'])

    Args:
        obj: Either a dictionary or an object with attributes
        key: The key/attribute name to retrieve
        default: Default value if key not found

    Returns:
        The value associated with key, or default if not found
    """
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


# =============================================================================
# PYDANTIC MODELS
# =============================================================================
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


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def _format_bet(bet: Any, game_id_override: str = None) -> dict:
    """
    Format a bet object into a dictionary matching the ValueBet frontend interface.

    Uses get_val() to handle both Object and Dictionary data types.

    Args:
        bet: The bet object (can be ValueBet, FallbackValueBet, dict, or similar)
        game_id_override: Optional game_id to use instead of bet's game_id

    Returns:
        Dictionary with all ValueBet fields
    """
    # Get urgency value - handle enum or string
    urgency_val = get_val(bet, "urgency", "medium")
    if hasattr(urgency_val, "value"):
        urgency_str = urgency_val.value
    else:
        urgency_str = str(urgency_val)

    # Get detected_at - handle datetime or None
    detected_at = get_val(bet, "detected_at", None)
    if detected_at is not None:
        if isinstance(detected_at, datetime):
            detected_at_str = detected_at.isoformat()
        else:
            detected_at_str = str(detected_at)
    else:
        detected_at_str = datetime.now().isoformat()

    # Get expires_at - handle datetime or None
    expires_at = get_val(bet, "expires_at", None)
    if expires_at is not None:
        if isinstance(expires_at, datetime):
            expires_at_str = expires_at.isoformat()
        else:
            expires_at_str = str(expires_at)
    else:
        expires_at_str = None

    # Extract game_id using universal accessor
    bet_game_id = get_val(bet, "game_id", "")

    return {
        "game_id": game_id_override or bet_game_id,
        "bet_type": get_val(bet, "bet_type", ""),
        "description": get_val(bet, "description", ""),
        "model_probability": get_val(bet, "model_probability", 0),
        "model_prediction": get_val(bet, "model_prediction", 0),
        "bookmaker": get_val(bet, "bookmaker", ""),
        "odds": get_val(bet, "odds", 0),
        "implied_probability": get_val(bet, "implied_probability", 0),
        "line": get_val(bet, "line", 0),
        "edge": get_val(bet, "edge", 0),
        "expected_value": get_val(bet, "expected_value", 0),
        "recommended_stake": get_val(bet, "recommended_stake", None),
        "urgency": urgency_str,
        "detected_at": detected_at_str,
        "expires_at": expires_at_str,
    }


def _get_bet_game_id(bet: Any) -> str:
    """
    Extract game_id from a bet using universal accessor.

    Args:
        bet: The bet object (can be Object or Dictionary)

    Returns:
        The game_id as a string
    """
    return str(get_val(bet, "game_id", ""))


def _get_bet_edge(bet: Any) -> float:
    """
    Extract edge from a bet using universal accessor.

    Args:
        bet: The bet object (can be Object or Dictionary)

    Returns:
        The edge as a float
    """
    edge = get_val(bet, "edge", 0)
    try:
        return float(edge)
    except (TypeError, ValueError):
        return 0.0


def _game_ids_match(bet_game_id: str, query_game_id: str) -> bool:
    """
    Match game_ids even if season differs (2024 vs 2025).

    Game ID format: YYYY_WW_AWAY_HOME (e.g., "2024_17_HOU_LAC")

    This handles the common case where:
    - Games from nflverse use one season year
    - Value bets use a different season year
    - But week + away + home teams are the same

    Args:
        bet_game_id: The game_id from the bet
        query_game_id: The game_id being searched for

    Returns:
        True if the game IDs match (exact or fuzzy)
    """
    # Exact match - fast path
    if bet_game_id == query_game_id:
        return True

    # Fuzzy match - compare week + teams, ignore season
    try:
        bet_parts = bet_game_id.split("_")
        query_parts = query_game_id.split("_")

        # Both must have at least 4 parts: YYYY_WW_AWAY_HOME
        if len(bet_parts) >= 4 and len(query_parts) >= 4:
            # Compare week, away team, home team (indices 1, 2, 3)
            return bet_parts[1:4] == query_parts[1:4]
    except Exception:
        pass

    return False


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

    # Group bets by game using universal accessor
    bets_by_game = defaultdict(list)
    for bet in fallback_bets:
        game_id = _get_bet_game_id(bet)
        if game_id:
            bets_by_game[game_id].append(bet)

    for game in fallback_games:
        game_id = get_val(game, "game_id", "")
        game_bets = bets_by_game.get(game_id, [])

        # Find best edge bet using universal accessor
        best_bet = None
        if game_bets:
            best_bet = max(game_bets, key=_get_bet_edge)

        response_games.append(
            GameResponse(
                game_id=game_id,
                home_team=get_val(game, "home_team", ""),
                away_team=get_val(game, "away_team", ""),
                kickoff=get_val(game, "kickoff", ""),
                week=get_val(game, "week", 17),
                season=get_val(game, "season", 2024),
                value_bet_count=len(game_bets),
                best_edge=get_val(best_bet, "edge", None) if best_bet else None,
                best_bet_description=get_val(best_bet, "description", None) if best_bet else None,
                model_prediction=get_val(best_bet, "model_prediction", None) if best_bet else None,
                model_confidence=get_val(best_bet, "model_probability", None) if best_bet else None,
                vegas_line=get_val(game, "spread", None),
            )
        )

    # Sort by kickoff time, then by value bet count (most bets first)
    response_games.sort(key=lambda g: (g.kickoff, -g.value_bet_count))

    return response_games


# =============================================================================
# API ENDPOINTS
# =============================================================================
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
    is_initializing = get_val(app_state, "_is_initializing", True)

    # COLD START: If still initializing and no data, tell frontend to retry
    if is_initializing and not app_state.last_value_bets:
        # Check if we have cached data to show
        cached_bets = get_val(app_state, "_cached_bets_raw", None)
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

    # Get value bets and group by game using universal accessor
    value_bets = app_state.last_value_bets
    bets_by_game = defaultdict(list)
    for bet in value_bets:
        game_id = _get_bet_game_id(bet)
        if game_id:
            bets_by_game[game_id].append(bet)

    # Build response
    response_games = []

    for game in games_data:
        game_id = get_val(game, "game_id", "")
        game_bets = bets_by_game.get(game_id, [])

        # Find best edge bet using universal accessor
        best_bet = None
        if game_bets:
            best_bet = max(game_bets, key=_get_bet_edge)

        # Extract Vegas line from odds if available
        vegas_line = None
        odds = get_val(game, "odds", None)
        if odds and isinstance(odds, dict):
            bookmakers = odds.get("bookmakers", [])
            for book in bookmakers:
                markets = book.get("markets", [])
                for market in markets:
                    if market.get("key") == "spreads":
                        outcomes = market.get("outcomes", [])
                        for outcome in outcomes:
                            if outcome.get("name") == get_val(game, "home_team", ""):
                                vegas_line = outcome.get("point")
                                break
                if vegas_line is not None:
                    break

        # Get model prediction from best bet if available
        model_prediction = None
        model_confidence = None
        if best_bet:
            model_prediction = get_val(best_bet, "model_prediction", None)
            model_confidence = get_val(best_bet, "model_probability", None)

        # Filter by week if specified
        game_week = get_val(game, "week", None)
        if week is not None and game_week != week:
            continue

        # Format kickoff time
        kickoff = get_val(game, "kickoff", None)
        kickoff_str = kickoff.isoformat() if isinstance(kickoff, datetime) else str(kickoff or "")

        response_games.append(
            GameResponse(
                game_id=game_id,
                home_team=get_val(game, "home_team", ""),
                away_team=get_val(game, "away_team", ""),
                kickoff=kickoff_str,
                week=get_val(game, "week", 17),
                season=get_val(game, "season", 2024),
                value_bet_count=len(game_bets),
                best_edge=get_val(best_bet, "edge", None) if best_bet else None,
                best_bet_description=get_val(best_bet, "description", None) if best_bet else None,
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
    Uses ONLY REAL DATA from nflverse and The Odds API.
    """
    app_state = request.app.state.app_state

    # Check if system is initialized
    is_initialized = get_val(app_state, "_initialized", False)

    if not is_initialized:
        logger.info(f"System not initialized - returning 503 for game {game_id}")
        return {
            "error": "System initializing",
            "game_id": game_id,
            "is_demo": False,
            "is_initializing": True,
            "retry_after_seconds": 5,
        }

    # Get game from pipeline (REAL DATA only)
    game_data = None
    if app_state.pipeline:
        try:
            enriched_games = await app_state.pipeline.get_upcoming_games_enriched(
                include_odds=True,
                include_dvoa=False,
                include_health=False,
            )
            for game in enriched_games:
                if get_val(game, "game_id", "") == game_id:
                    game_data = game
                    break
        except Exception as e:
            logger.warning(f"Pipeline query failed for game {game_id}: {e}")

    # Get value bets for this game using FUZZY matching (handles season mismatch)
    value_bets = app_state.last_value_bets

    # ==========================================================================
    # DEBUG: Type mismatch detection - this will appear in Railway logs
    # ==========================================================================
    target_game_id = str(game_id)  # Normalize to string
    print(f"DEBUG: Request game_id='{target_game_id}' (Type: {type(game_id).__name__})")
    print(f"DEBUG: Total value_bets in memory: {len(value_bets)}")

    # Debug logging - show what we're trying to match with types
    for i, bet in enumerate(value_bets[:5]):
        bet_game_id = str(get_val(bet, "game_id", ""))
        print(f"DEBUG: Bet[{i}] game_id='{bet_game_id}' (Type: {type(get_val(bet, 'game_id', '')).__name__}) | Match: {_game_ids_match(bet_game_id, target_game_id)}")

    bet_game_ids = [_get_bet_game_id(bet) for bet in value_bets[:5]]
    logger.info(f"[{game_id}] Looking for match in bet game_ids: {bet_game_ids}")

    # Filter bets using FUZZY matching with normalized string comparison
    game_bets = []
    for bet in value_bets:
        bet_game_id = str(get_val(bet, "game_id", ""))
        if _game_ids_match(bet_game_id, target_game_id):
            game_bets.append(bet)

    print(f"DEBUG: Found {len(game_bets)} matching bets for game '{target_game_id}'")
    logger.info(f"[{game_id}] Found {len(game_bets)} matching bets using fuzzy matching")

    # =========================================================================
    # FALLBACK RECOVERY: If pipeline failed, reconstruct game from bets or fallback data
    # =========================================================================
    is_fallback = False
    if not game_data:
        logger.warning(f"🔄 Pipeline returned no data for game {game_id} - attempting fallback recovery")

        # First, try to find game in fallback games using FUZZY matching
        fallback_games = app_state.get_fallback_games()
        for fg in fallback_games:
            if _game_ids_match(get_val(fg, "game_id", ""), game_id):
                logger.info(f"✅ Found game {game_id} in fallback games")
                # Create synthetic game object from fallback data
                game_data = type("SyntheticGame", (), {
                    "game_id": game_id,
                    "home_team": get_val(fg, "home_team", "Unknown"),
                    "away_team": get_val(fg, "away_team", "Unknown"),
                    "kickoff": get_val(fg, "kickoff", datetime.now().isoformat()),
                    "week": get_val(fg, "week", 17),
                    "season": get_val(fg, "season", 2024),
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

            # Try to extract kickoff from bet if available using universal accessor
            sample_bet = game_bets[0]
            kickoff = get_val(sample_bet, "detected_at", datetime.now())
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

    # Format kickoff using universal accessor
    kickoff_val = get_val(game_data, "kickoff", datetime.now())
    kickoff_str = kickoff_val.isoformat() if isinstance(kickoff_val, datetime) else str(kickoff_val)

    return {
        "game_id": game_id,
        "home_team": get_val(game_data, "home_team", "Unknown"),
        "away_team": get_val(game_data, "away_team", "Unknown"),
        "kickoff": kickoff_str,
        "week": get_val(game_data, "week", 17),
        "season": get_val(game_data, "season", 2024),
        "value_bets": formatted_bets,
        "value_bet_count": len(formatted_bets),
        "is_demo": False,
        "is_fallback": is_fallback,
    }
