"""Raw predictions endpoints - model predictions independent of betting lines."""

import logging
from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel

router = APIRouter()
logger = logging.getLogger(__name__)


class PropPredictionResponse(BaseModel):
    """Response model for a player prop prediction."""

    player_name: str
    team: str
    opponent: str
    game_id: str
    prop_type: str
    predicted_value: float
    range_low: float  # 25th percentile
    range_high: float  # 75th percentile
    confidence: float  # Based on prediction std
    injury_status: Optional[str] = None  # ACTIVE, QUESTIONABLE, OUT, etc.
    # DraftKings line info (if available)
    dk_line: Optional[float] = None  # e.g., 250.5
    dk_over_odds: Optional[int] = None  # e.g., -110
    dk_under_odds: Optional[int] = None  # e.g., -110
    # Betting recommendation (if line available)
    recommendation: Optional[str] = None  # "OVER", "UNDER", or None
    edge: Optional[float] = None  # Edge percentage (e.g., 0.05 = 5%)
    bet_confidence: Optional[str] = None  # "LOW", "MEDIUM", "HIGH"


class InjuryAdjustedPredictionResponse(BaseModel):
    """Response model for injury-adjusted player prop prediction."""

    player_name: str
    team: str
    opponent: str
    game_id: str
    prop_type: str
    predicted_value: float
    range_low: float  # 25th percentile (widened by uncertainty)
    range_high: float  # 75th percentile (widened by uncertainty)
    confidence: float
    injury_status: str
    # Injury context
    uncertainty_multiplier: float = 1.0
    is_backup_starter: bool = False
    replacing_player: Optional[str] = None
    usage_boost_applied: bool = False
    usage_boost_reason: Optional[str] = None


class BackupPlayerPrediction(BaseModel):
    """Prediction for a backup player stepping into starter role."""

    player_name: str
    team: str
    opponent: str
    game_id: str
    prop_type: str
    predicted_value: float
    range_low: float
    range_high: float
    confidence: float
    injury_status: str
    is_backup: bool = True
    replacing_player: str
    replacing_injury_status: str
    uncertainty_note: str


class EnhancedGamePredictionsResponse(BaseModel):
    """Game predictions including backup players and injury context."""

    game_id: str
    home_team: str
    away_team: str
    kickoff: str
    spread_prediction: Optional[float] = None
    player_props: list[InjuryAdjustedPredictionResponse]
    backup_player_props: list[BackupPlayerPrediction]
    injury_summary: dict  # Summary of key injuries affecting predictions


class SpreadPredictionResponse(BaseModel):
    """Response model for spread prediction."""

    predicted_spread: float  # Model's predicted spread (negative = home favored)
    prediction_std: float  # Uncertainty in prediction
    confidence_low: float  # 25th percentile
    confidence_high: float  # 75th percentile
    home_cover_prob: float  # Probability home team covers
    away_cover_prob: float  # Probability away team covers
    # DraftKings line info
    dk_line: Optional[float] = None
    dk_home_odds: Optional[int] = None
    dk_away_odds: Optional[int] = None
    # Betting recommendation
    recommendation: Optional[str] = None  # "HOME", "AWAY", or None
    edge: Optional[float] = None  # Edge percentage
    bet_confidence: Optional[str] = None  # "LOW", "MEDIUM", "HIGH"


class GamePredictionsResponse(BaseModel):
    """Response model for predictions for a single game."""

    game_id: str
    home_team: str
    away_team: str
    kickoff: str
    spread_prediction: Optional[float] = None  # Deprecated: use spread field
    spread: Optional[SpreadPredictionResponse] = None  # Full spread prediction
    player_props: list[PropPredictionResponse]


class PredictionsListResponse(BaseModel):
    """Response model for all predictions."""

    count: int
    games: list[GamePredictionsResponse]
    generated_at: str


# Key players by team - QB, top RB, top WR for each team
# This is a simplified approach - ideally we'd pull from nflverse rosters
KEY_PLAYERS = {
    "KC": [
        {"name": "Patrick Mahomes", "position": "QB"},
        {"name": "Isiah Pacheco", "position": "RB"},
        {"name": "Travis Kelce", "position": "TE"},
    ],
    "BUF": [
        {"name": "Josh Allen", "position": "QB"},
        {"name": "James Cook", "position": "RB"},
        {"name": "Khalil Shakir", "position": "WR"},
    ],
    "BAL": [
        {"name": "Lamar Jackson", "position": "QB"},
        {"name": "Derrick Henry", "position": "RB"},
        {"name": "Zay Flowers", "position": "WR"},
    ],
    "DET": [
        {"name": "Jared Goff", "position": "QB"},
        {"name": "Jahmyr Gibbs", "position": "RB"},
        {"name": "Amon-Ra St. Brown", "position": "WR"},
    ],
    "PHI": [
        {"name": "Jalen Hurts", "position": "QB"},
        {"name": "Saquon Barkley", "position": "RB"},
        {"name": "A.J. Brown", "position": "WR"},
    ],
    "MIN": [
        {"name": "Sam Darnold", "position": "QB"},
        {"name": "Aaron Jones", "position": "RB"},
        {"name": "Justin Jefferson", "position": "WR"},
    ],
    "GB": [
        {"name": "Jordan Love", "position": "QB"},
        {"name": "Josh Jacobs", "position": "RB"},
        {"name": "Jayden Reed", "position": "WR"},
    ],
    "LAR": [
        {"name": "Matthew Stafford", "position": "QB"},
        {"name": "Kyren Williams", "position": "RB"},
        {"name": "Puka Nacua", "position": "WR"},
    ],
    "WAS": [
        {"name": "Jayden Daniels", "position": "QB"},
        {"name": "Brian Robinson Jr.", "position": "RB"},
        {"name": "Terry McLaurin", "position": "WR"},
    ],
    "SF": [
        {"name": "Brock Purdy", "position": "QB"},
        {"name": "Christian McCaffrey", "position": "RB"},
        {"name": "Deebo Samuel", "position": "WR"},
    ],
    "DAL": [
        {"name": "Cooper Rush", "position": "QB"},
        {"name": "Rico Dowdle", "position": "RB"},
        {"name": "CeeDee Lamb", "position": "WR"},
    ],
    "CIN": [
        {"name": "Joe Burrow", "position": "QB"},
        {"name": "Chase Brown", "position": "RB"},
        {"name": "Ja'Marr Chase", "position": "WR"},
    ],
    "MIA": [
        {"name": "Tua Tagovailoa", "position": "QB"},
        {"name": "De'Von Achane", "position": "RB"},
        {"name": "Tyreek Hill", "position": "WR"},
    ],
    "DEN": [
        {"name": "Bo Nix", "position": "QB"},
        {"name": "Javonte Williams", "position": "RB"},
        {"name": "Courtland Sutton", "position": "WR"},
    ],
    "LAC": [
        {"name": "Justin Herbert", "position": "QB"},
        {"name": "J.K. Dobbins", "position": "RB"},
        {"name": "Ladd McConkey", "position": "WR"},
    ],
    "PIT": [
        {"name": "Russell Wilson", "position": "QB"},
        {"name": "Najee Harris", "position": "RB"},
        {"name": "George Pickens", "position": "WR"},
    ],
    "HOU": [
        {"name": "C.J. Stroud", "position": "QB"},
        {"name": "Joe Mixon", "position": "RB"},
        {"name": "Nico Collins", "position": "WR"},
    ],
    "IND": [
        {"name": "Anthony Richardson", "position": "QB"},
        {"name": "Jonathan Taylor", "position": "RB"},
        {"name": "Michael Pittman Jr.", "position": "WR"},
    ],
    "ARI": [
        {"name": "Kyler Murray", "position": "QB"},
        {"name": "James Conner", "position": "RB"},
        {"name": "Marvin Harrison Jr.", "position": "WR"},
    ],
    "ATL": [
        {"name": "Kirk Cousins", "position": "QB"},
        {"name": "Bijan Robinson", "position": "RB"},
        {"name": "Drake London", "position": "WR"},
    ],
    "SEA": [
        {"name": "Geno Smith", "position": "QB"},
        {"name": "Kenneth Walker III", "position": "RB"},
        {"name": "DK Metcalf", "position": "WR"},
    ],
    "TB": [
        {"name": "Baker Mayfield", "position": "QB"},
        {"name": "Rachaad White", "position": "RB"},
        {"name": "Mike Evans", "position": "WR"},
    ],
    "NO": [
        {"name": "Derek Carr", "position": "QB"},
        {"name": "Alvin Kamara", "position": "RB"},
        {"name": "Chris Olave", "position": "WR"},
    ],
    "CHI": [
        {"name": "Caleb Williams", "position": "QB"},
        {"name": "D'Andre Swift", "position": "RB"},
        {"name": "DJ Moore", "position": "WR"},
    ],
    "CLE": [
        {"name": "Jameis Winston", "position": "QB"},
        {"name": "Nick Chubb", "position": "RB"},
        {"name": "Jerry Jeudy", "position": "WR"},
    ],
    "JAX": [
        {"name": "Mac Jones", "position": "QB"},
        {"name": "Travis Etienne Jr.", "position": "RB"},
        {"name": "Brian Thomas Jr.", "position": "WR"},
    ],
    "NYG": [
        {"name": "Tommy DeVito", "position": "QB"},
        {"name": "Tyrone Tracy Jr.", "position": "RB"},
        {"name": "Malik Nabers", "position": "WR"},
    ],
    "NYJ": [
        {"name": "Aaron Rodgers", "position": "QB"},
        {"name": "Breece Hall", "position": "RB"},
        {"name": "Garrett Wilson", "position": "WR"},
    ],
    "LV": [
        {"name": "Aidan O'Connell", "position": "QB"},
        {"name": "Alexander Mattison", "position": "RB"},
        {"name": "Jakobi Meyers", "position": "WR"},
    ],
    "NE": [
        {"name": "Drake Maye", "position": "QB"},
        {"name": "Rhamondre Stevenson", "position": "RB"},
        {"name": "Demario Douglas", "position": "WR"},
    ],
    "CAR": [
        {"name": "Bryce Young", "position": "QB"},
        {"name": "Chuba Hubbard", "position": "RB"},
        {"name": "Adam Thielen", "position": "WR"},
    ],
    "TEN": [
        {"name": "Will Levis", "position": "QB"},
        {"name": "Tony Pollard", "position": "RB"},
        {"name": "Calvin Ridley", "position": "WR"},
    ],
}


def _get_prop_type_for_position(position: str) -> list[str]:
    """Get appropriate prop types for a position."""
    if position == "QB":
        return ["passing_yards"]
    elif position == "RB":
        return ["rushing_yards"]
    elif position in ("WR", "TE"):
        return ["receiving_yards"]
    return []


# Team name mapping from full names to abbreviations (for Odds API)
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

# Odds API prop market to internal prop type mapping
PROP_MARKET_MAP = {
    "player_pass_yds": "passing_yards",
    "player_rush_yds": "rushing_yards",
    "player_reception_yds": "receiving_yards",
    "player_receptions": "receptions",
}


async def _fetch_draftkings_lines(
    app_state: Any,
    home_team: str,
    away_team: str,
) -> dict[str, dict]:
    """
    Fetch DraftKings prop lines for a game.

    Returns dict mapping (player_name_lower, prop_type) -> line_info
    """
    lines = {}

    if not hasattr(app_state, 'pipeline') or not app_state.pipeline:
        return lines

    odds_api = getattr(app_state.pipeline, 'odds_api', None)
    if not odds_api:
        logger.debug("No odds_api client available")
        return lines

    try:
        # Get Odds API events to find event_id for this game
        events = await odds_api.get_events()
        if not events:
            logger.debug("No events from Odds API")
            return lines

        # Find the event matching our game
        event_id = None
        for event in events:
            event_home = TEAM_NAME_MAP.get(event.get("home_team", ""), "")
            event_away = TEAM_NAME_MAP.get(event.get("away_team", ""), "")
            if event_home == home_team and event_away == away_team:
                event_id = event.get("id")
                break

        if not event_id:
            logger.debug(f"No Odds API event found for {away_team} @ {home_team}")
            return lines

        # Fetch player props from DraftKings
        props_data = await odds_api.get_player_props(
            event_id=event_id,
            bookmakers=["draftkings"],
        )

        if not props_data or "bookmakers" not in props_data:
            logger.debug(f"No props data for event {event_id}")
            return lines

        # Parse the response
        for bookmaker in props_data.get("bookmakers", []):
            if bookmaker.get("key") != "draftkings":
                continue

            for market in bookmaker.get("markets", []):
                market_key = market.get("key", "")
                prop_type = PROP_MARKET_MAP.get(market_key)
                if not prop_type:
                    continue

                # Group outcomes by player (Over/Under pairs)
                player_outcomes = {}
                for outcome in market.get("outcomes", []):
                    player_name = outcome.get("description", "")
                    if not player_name:
                        continue

                    if player_name not in player_outcomes:
                        player_outcomes[player_name] = {}

                    side = outcome.get("name", "").lower()
                    player_outcomes[player_name][side] = {
                        "point": outcome.get("point"),
                        "price": outcome.get("price"),
                    }

                # Store parsed lines
                for player_name, outcomes in player_outcomes.items():
                    over_data = outcomes.get("over", {})
                    under_data = outcomes.get("under", {})

                    if over_data.get("point") is not None:
                        key = (player_name.lower().strip(), prop_type)
                        lines[key] = {
                            "line": over_data.get("point"),
                            "over_odds": over_data.get("price"),
                            "under_odds": under_data.get("price"),
                        }

        logger.info(f"Fetched {len(lines)} DraftKings lines for {away_team} @ {home_team}")

    except Exception as e:
        logger.warning(f"Failed to fetch DraftKings lines: {e}")

    return lines


async def _fetch_draftkings_spread(
    app_state: Any,
    home_team: str,
    away_team: str,
) -> Optional[dict]:
    """
    Fetch DraftKings spread line for a game.

    Returns dict with line, home_odds, away_odds or None if not available.
    """
    if not hasattr(app_state, 'pipeline') or not app_state.pipeline:
        return None

    try:
        # Get odds data from pipeline (includes spreads)
        odds_data = await app_state.pipeline.get_game_odds(markets=["spreads"])
        if not odds_data:
            logger.debug("No odds data available from pipeline")
            return None

        # Find the event matching our game
        for event in odds_data:
            event_home = TEAM_NAME_MAP.get(event.get("home_team", ""), "")
            event_away = TEAM_NAME_MAP.get(event.get("away_team", ""), "")

            if event_home != home_team or event_away != away_team:
                continue

            # Found the game - look for DraftKings spread
            for bookmaker in event.get("bookmakers", []):
                if bookmaker.get("key") != "draftkings":
                    continue

                for market in bookmaker.get("markets", []):
                    if market.get("key") != "spreads":
                        continue

                    outcomes = market.get("outcomes", [])
                    home_spread = None
                    home_odds = None
                    away_odds = None

                    for outcome in outcomes:
                        team_name = TEAM_NAME_MAP.get(outcome.get("name", ""), "")
                        if team_name == home_team:
                            home_spread = outcome.get("point")
                            home_odds = outcome.get("price")
                        elif team_name == away_team:
                            away_odds = outcome.get("price")

                    if home_spread is not None:
                        logger.debug(f"Found DK spread for {away_team}@{home_team}: {home_spread}")
                        return {
                            "line": home_spread,
                            "home_odds": home_odds,
                            "away_odds": away_odds,
                        }

        logger.debug(f"No DK spread found for {away_team}@{home_team}")

    except Exception as e:
        logger.warning(f"Failed to fetch DraftKings spread: {e}")

    return None


def _calculate_spread_edge(
    predicted_spread: float,
    vegas_line: float,
    home_cover_prob: float,
) -> tuple[Optional[str], Optional[float], Optional[str]]:
    """
    Calculate spread betting recommendation.

    Args:
        predicted_spread: Model's predicted spread (negative = home favored)
        vegas_line: Vegas/DK spread line (negative = home favored)
        home_cover_prob: Probability home team covers the spread

    Returns (recommendation, edge, confidence)
    - recommendation: "HOME", "AWAY", or None
    - edge: edge as probability (e.g., 0.08 = 8%)
    - confidence: "LOW", "MEDIUM", "HIGH"
    """
    if vegas_line is None:
        return None, None, None

    # Calculate edge based on cover probability vs implied 50%
    # If home_cover_prob > 0.5, bet HOME; else bet AWAY
    if home_cover_prob > 0.5:
        recommendation = "HOME"
        edge = home_cover_prob - 0.5  # Edge over 50/50
    else:
        recommendation = "AWAY"
        edge = (1 - home_cover_prob) - 0.5

    # Only recommend if edge is significant (> 3%)
    if edge < 0.03:
        return None, edge, "LOW"

    # Determine confidence based on edge size
    if edge >= 0.10:  # 10%+ edge
        confidence = "HIGH"
    elif edge >= 0.05:  # 5-10% edge
        confidence = "MEDIUM"
    else:  # 3-5% edge
        confidence = "LOW"

    return recommendation, edge, confidence


def _calculate_betting_edge(
    predicted_value: float,
    line: float,
    prediction_std: float,
) -> tuple[Optional[str], Optional[float], Optional[str]]:
    """
    Calculate betting recommendation based on prediction vs line.

    Returns (recommendation, edge, confidence)
    - recommendation: "OVER", "UNDER", or None
    - edge: percentage edge (e.g., 0.08 = 8%)
    - confidence: "LOW", "MEDIUM", "HIGH"
    """
    if line is None or line <= 0:
        return None, None, None

    # Calculate how many std devs away the line is from prediction
    diff = predicted_value - line
    diff_pct = diff / line

    # Calculate edge as percentage difference
    edge = abs(diff_pct)

    # Determine recommendation
    if diff > 0:
        recommendation = "OVER"
    else:
        recommendation = "UNDER"

    # Only recommend if edge is significant (> 3%)
    if edge < 0.03:
        return None, edge, "LOW"

    # Determine confidence based on edge size and prediction confidence
    if edge >= 0.10:  # 10%+ edge
        confidence = "HIGH"
    elif edge >= 0.05:  # 5-10% edge
        confidence = "MEDIUM"
    else:  # 3-5% edge
        confidence = "LOW"

    return recommendation, edge, confidence


@router.get("/predictions", response_model=PredictionsListResponse)
async def get_all_predictions(
    request: Request,
    game_id: Optional[str] = Query(None, description="Filter to specific game"),
) -> dict[str, Any]:
    """
    Get raw player prop predictions for upcoming games.

    Returns predictions like:
    - Patrick Mahomes: 285 passing yards (range: 210-360)
    - Derrick Henry: 89 rushing yards (range: 55-125)

    These are model predictions independent of betting lines.
    """
    app_state = request.app.state.app_state

    if not app_state.pipeline or not app_state.feature_pipeline:
        return {
            "count": 0,
            "games": [],
            "generated_at": datetime.now().isoformat(),
            "error": "Pipeline not initialized",
        }

    # Get value detector with prop models
    value_detector = app_state.value_detector
    if not value_detector or not value_detector.prop_models:
        return {
            "count": 0,
            "games": [],
            "generated_at": datetime.now().isoformat(),
            "error": "Prop models not loaded",
        }

    # Get upcoming games
    try:
        games_df = await app_state.pipeline.get_upcoming_games()
        if games_df is None or len(games_df) == 0:
            return {
                "count": 0,
                "games": [],
                "generated_at": datetime.now().isoformat(),
            }
        games = games_df.to_dicts()
    except Exception as e:
        logger.error(f"Failed to get upcoming games: {e}")
        return {
            "count": 0,
            "games": [],
            "generated_at": datetime.now().isoformat(),
            "error": str(e),
        }

    # Filter to specific game if requested
    if game_id:
        games = [g for g in games if g.get("game_id") == game_id]

    # Generate predictions for each game
    game_predictions = []
    total_players_processed = 0
    total_players_found = 0

    logger.info(f"Generating predictions for {len(games)} games")

    for game in games:
        gid = game.get("game_id", "")
        home_team = game.get("home_team", "")
        away_team = game.get("away_team", "")
        season = game.get("season", 2025)
        week = game.get("week", 1)
        kickoff = game.get("commence_time") or game.get("gameday") or ""

        if isinstance(kickoff, datetime):
            kickoff = kickoff.isoformat()

        # Generate spread prediction using spread model
        spread_response = None
        spread_pred = None  # Legacy field

        spread_model = value_detector.spread_model
        if spread_model:
            try:
                # Build spread features
                spread_features = await app_state.feature_pipeline.build_spread_features(
                    game_id=gid,
                    home_team=home_team,
                    away_team=away_team,
                    season=season,
                    week=int(week),
                )

                if spread_features and spread_features.features:
                    # Get spread prediction
                    spread_prediction = spread_model.predict_game(
                        features=spread_features.features,
                        game_id=gid,
                        home_team=home_team,
                        away_team=away_team,
                        line=None,  # No line yet, will add DK line below
                    )

                    spread_pred = round(spread_prediction.predicted_spread, 1)

                    # Fetch DraftKings spread line
                    dk_spread = await _fetch_draftkings_spread(app_state, home_team, away_team)

                    # Calculate betting recommendation if DK line available
                    recommendation = None
                    edge = None
                    bet_confidence = None

                    if dk_spread:
                        recommendation, edge, bet_confidence = _calculate_spread_edge(
                            predicted_spread=spread_prediction.predicted_spread,
                            vegas_line=dk_spread.get("line"),
                            home_cover_prob=spread_prediction.home_cover_prob,
                        )

                    spread_response = SpreadPredictionResponse(
                        predicted_spread=round(spread_prediction.predicted_spread, 1),
                        prediction_std=round(spread_prediction.prediction_std, 2),
                        confidence_low=round(spread_prediction.confidence_lower, 1),
                        confidence_high=round(spread_prediction.confidence_upper, 1),
                        home_cover_prob=round(spread_prediction.home_cover_prob, 3),
                        away_cover_prob=round(spread_prediction.away_cover_prob, 3),
                        dk_line=dk_spread.get("line") if dk_spread else None,
                        dk_home_odds=dk_spread.get("home_odds") if dk_spread else None,
                        dk_away_odds=dk_spread.get("away_odds") if dk_spread else None,
                        recommendation=recommendation,
                        edge=round(edge, 4) if edge is not None else None,
                        bet_confidence=bet_confidence,
                    )

                    logger.info(f"Spread prediction for {gid}: {spread_pred} (DK: {dk_spread.get('line') if dk_spread else 'N/A'})")

            except Exception as e:
                logger.warning(f"Failed to generate spread prediction for {gid}: {e}")

        # Fetch DraftKings prop lines for this game
        dk_lines = await _fetch_draftkings_lines(app_state, home_team, away_team)
        logger.info(f"Got {len(dk_lines)} DraftKings prop lines for {away_team} @ {home_team}")

        # Get player predictions for both teams
        player_preds = []

        for team, opponent in [(home_team, away_team), (away_team, home_team)]:
            # Try dynamic roster from ESPN first, fall back to static KEY_PLAYERS
            team_players = []
            if app_state.pipeline:
                try:
                    team_players = await app_state.pipeline.get_key_players_dynamic(
                        team=team,
                        season=season,
                        positions=["QB", "RB", "WR", "TE"],
                        max_per_position=1,
                        exclude_injured=False,  # Include injured players, show status
                    )
                except Exception as e:
                    logger.debug(f"ESPN roster lookup failed for {team}: {e}")

            # Fall back to static roster if ESPN fails
            if not team_players:
                static_players = KEY_PLAYERS.get(team, [])
                team_players = [
                    {"name": p["name"], "position": p["position"], "injury_status": "UNKNOWN"}
                    for p in static_players
                ]

            for player_info in team_players:
                player_name = player_info.get("name", "")
                position = player_info.get("position", "")
                injury_status = player_info.get("injury_status", "ACTIVE")
                prop_types = _get_prop_type_for_position(position)
                total_players_processed += 1

                # Look up the real nflverse player_id from name
                logger.info(f"Looking up player ID for: {player_name} (season={season}, pos={position})")
                player_id = await app_state.feature_pipeline.lookup_player_id(
                    player_name=player_name,
                    season=season,
                    position=position,
                )

                # Skip if player not found
                if not player_id:
                    logger.warning(f"Player ID NOT FOUND for {player_name} (season={season}) - skipping this player")
                    continue

                logger.info(f"Found player ID for {player_name}: {player_id}")
                total_players_found += 1

                for prop_type in prop_types:
                    # Check if we have this prop model
                    prop_model = value_detector.prop_models.get(prop_type)
                    if not prop_model:
                        continue

                    try:
                        # Build features for this player using real player_id
                        features = await app_state.feature_pipeline.build_prop_features(
                            game_id=gid,
                            player_id=player_id,  # Use real nflverse ID
                            player_name=player_name,
                            prop_type=prop_type,
                            season=season,
                            week=week,
                            opponent_team=opponent,
                            position=position,
                        )

                        if features is None or features.features is None:
                            continue

                        # Make prediction
                        prediction = prop_model.predict_player(
                            features=features.features,
                            player_id=player_id,  # Use real nflverse ID
                            player_name=player_name,
                            game_id=gid,
                            team=team,
                            opponent=opponent,
                            line=None,  # No line - raw prediction
                        )

                        # Calculate confidence from std dev
                        confidence = max(0.5, min(0.95, 1.0 - (prediction.prediction_std / prediction.predicted_value) if prediction.predicted_value > 0 else 0.5))

                        # Look up DraftKings line for this player/prop
                        dk_line_info = dk_lines.get((player_name.lower().strip(), prop_type))
                        dk_line = dk_line_info.get("line") if dk_line_info else None
                        dk_over_odds = dk_line_info.get("over_odds") if dk_line_info else None
                        dk_under_odds = dk_line_info.get("under_odds") if dk_line_info else None

                        # Calculate betting recommendation if line available
                        recommendation = None
                        edge = None
                        bet_confidence = None
                        if dk_line is not None:
                            recommendation, edge, bet_confidence = _calculate_betting_edge(
                                predicted_value=prediction.predicted_value,
                                line=dk_line,
                                prediction_std=prediction.prediction_std,
                            )

                        player_preds.append(
                            PropPredictionResponse(
                                player_name=player_name,
                                team=team,
                                opponent=opponent,
                                game_id=gid,
                                prop_type=prop_type,
                                predicted_value=round(prediction.predicted_value, 1),
                                range_low=round(prediction.quantile_25, 1),
                                range_high=round(prediction.quantile_75, 1),
                                confidence=round(confidence, 2),
                                injury_status=injury_status,
                                # DraftKings line info
                                dk_line=dk_line,
                                dk_over_odds=dk_over_odds,
                                dk_under_odds=dk_under_odds,
                                # Betting recommendation
                                recommendation=recommendation,
                                edge=round(edge, 4) if edge is not None else None,
                                bet_confidence=bet_confidence,
                            )
                        )

                    except Exception as e:
                        logger.debug(f"Failed to predict {prop_type} for {player_name}: {e}")
                        continue

        # Sort by predicted value descending within each prop type
        player_preds.sort(key=lambda p: (-p.predicted_value, p.prop_type))

        game_predictions.append(
            GamePredictionsResponse(
                game_id=gid,
                home_team=home_team,
                away_team=away_team,
                kickoff=kickoff,
                spread_prediction=spread_pred,
                spread=spread_response,
                player_props=player_preds,
            )
        )

    logger.info(f"Prediction summary: {total_players_processed} players processed, {total_players_found} IDs found, {sum(len(g.player_props) for g in game_predictions)} predictions generated")

    return {
        "count": len(game_predictions),
        "games": game_predictions,
        "generated_at": datetime.now().isoformat(),
    }


@router.get("/predictions/debug/spread-odds")
async def debug_spread_odds(
    request: Request,
    home_team: str = Query("LAC", description="Home team abbreviation"),
    away_team: str = Query("HOU", description="Away team abbreviation"),
) -> dict[str, Any]:
    """
    Debug endpoint to test DraftKings spread fetching.
    """
    app_state = request.app.state.app_state

    result = {
        "home_team": home_team,
        "away_team": away_team,
        "pipeline_exists": app_state.pipeline is not None,
        "odds_data_count": 0,
        "matching_events": [],
        "dk_spread": None,
    }

    if not app_state.pipeline:
        return result

    try:
        odds_data = await app_state.pipeline.get_game_odds(markets=["spreads"])
        result["odds_data_count"] = len(odds_data) if odds_data else 0

        if odds_data:
            for event in odds_data:
                event_home_raw = event.get("home_team", "")
                event_away_raw = event.get("away_team", "")
                event_home = TEAM_NAME_MAP.get(event_home_raw, event_home_raw)
                event_away = TEAM_NAME_MAP.get(event_away_raw, event_away_raw)

                event_info = {
                    "raw_home": event_home_raw,
                    "raw_away": event_away_raw,
                    "mapped_home": event_home,
                    "mapped_away": event_away,
                    "matches": event_home == home_team and event_away == away_team,
                    "bookmakers_count": len(event.get("bookmakers", [])),
                }

                # Check for DK spreads
                for bookmaker in event.get("bookmakers", []):
                    if bookmaker.get("key") == "draftkings":
                        for market in bookmaker.get("markets", []):
                            if market.get("key") == "spreads":
                                event_info["dk_spread_outcomes"] = market.get("outcomes", [])

                result["matching_events"].append(event_info)

                if event_home == home_team and event_away == away_team:
                    dk_spread = await _fetch_draftkings_spread(app_state, home_team, away_team)
                    result["dk_spread"] = dk_spread

    except Exception as e:
        result["error"] = str(e)

    return result


@router.get("/predictions/debug/player-lookup")
async def debug_player_lookup(
    request: Request,
    player_name: str = Query(..., description="Player name to look up"),
    season: int = Query(2024, description="Season to search"),
    position: Optional[str] = Query(None, description="Position hint (QB, RB, WR, TE)"),
) -> dict[str, Any]:
    """
    Debug endpoint to test player ID lookup.
    """
    app_state = request.app.state.app_state

    if not app_state.feature_pipeline:
        return {"error": "Feature pipeline not initialized"}

    # Try looking up the player
    player_id = await app_state.feature_pipeline.lookup_player_id(
        player_name=player_name,
        season=season,
        position=position,
    )

    # Also check what PBP data is available
    pbp_info = {}
    if app_state.pipeline:
        try:
            pbp_df = await app_state.pipeline.get_historical_pbp([season])
            if pbp_df is not None:
                pbp_info["row_count"] = len(pbp_df)
                pbp_info["all_columns"] = list(pbp_df.columns)
                pbp_info["has_passer_player_name"] = "passer_player_name" in pbp_df.columns
                pbp_info["has_passer_id"] = "passer_id" in pbp_df.columns

                # Check for sample player names
                if "passer_player_name" in pbp_df.columns:
                    sample_passers = pbp_df.select("passer_player_name").unique().head(10).to_series().to_list()
                    pbp_info["sample_passers"] = [p for p in sample_passers if p is not None]

                    # Also show what we're searching for
                    parts = player_name.strip().split()
                    if len(parts) >= 2:
                        abbrev = f"{parts[0][0]}.{parts[-1]}"
                    else:
                        abbrev = player_name
                    pbp_info["searching_for_abbrev"] = abbrev.lower()
                    pbp_info["searching_for_full"] = player_name.lower()

                    # Try to find exact match
                    import polars as pl
                    exact_matches = pbp_df.filter(
                        pl.col("passer_player_name").str.to_lowercase() == abbrev.lower()
                    ).select("passer_id", "passer_player_name").unique().head(3).to_dicts()
                    pbp_info["exact_matches"] = exact_matches
        except Exception as e:
            pbp_info["error"] = str(e)

    return {
        "player_name": player_name,
        "season": season,
        "position": position,
        "player_id_found": player_id,
        "pbp_data": pbp_info,
    }


@router.get("/predictions/{game_id}")
async def get_game_predictions(
    request: Request,
    game_id: str,
) -> dict[str, Any]:
    """
    Get predictions for a specific game.

    Returns spread prediction and all player prop predictions.
    """
    # Don't call get_all_predictions() directly - FastAPI Query() params don't work on internal calls
    # Instead, use getPredictions with query param which works correctly
    app_state = request.app.state.app_state

    if not app_state.pipeline or not app_state.feature_pipeline:
        return {"error": "Pipeline not initialized", "game_id": game_id}

    value_detector = app_state.value_detector
    if not value_detector or not value_detector.prop_models:
        return {"error": "Prop models not loaded", "game_id": game_id}

    # Get upcoming games and filter to this specific game
    try:
        games_df = await app_state.pipeline.get_upcoming_games()
        if games_df is None or len(games_df) == 0:
            return {"error": "No upcoming games found", "game_id": game_id}
        games = games_df.to_dicts()
    except Exception as e:
        logger.error(f"Failed to get upcoming games: {e}")
        return {"error": str(e), "game_id": game_id}

    # Filter to specific game
    matching_games = [g for g in games if g.get("game_id") == game_id]
    if not matching_games:
        return {"error": "Game not found", "game_id": game_id}

    game = matching_games[0]
    gid = game.get("game_id", "")
    home_team = game.get("home_team", "")
    away_team = game.get("away_team", "")
    season = game.get("season", 2025)
    week = game.get("week", 1)
    kickoff = game.get("commence_time") or game.get("gameday") or ""

    if isinstance(kickoff, datetime):
        kickoff = kickoff.isoformat()

    # Generate spread prediction
    spread_response = None
    spread_pred = None

    spread_model = value_detector.spread_model
    if spread_model:
        try:
            spread_features = await app_state.feature_pipeline.build_spread_features(
                game_id=gid,
                home_team=home_team,
                away_team=away_team,
                season=season,
                week=int(week),
            )

            if spread_features and spread_features.features:
                spread_prediction = spread_model.predict_game(
                    features=spread_features.features,
                    game_id=gid,
                    home_team=home_team,
                    away_team=away_team,
                    line=None,
                )

                spread_pred = round(spread_prediction.predicted_spread, 1)

                dk_spread = await _fetch_draftkings_spread(app_state, home_team, away_team)

                recommendation = None
                edge = None
                bet_confidence = None

                if dk_spread:
                    recommendation, edge, bet_confidence = _calculate_spread_edge(
                        predicted_spread=spread_prediction.predicted_spread,
                        vegas_line=dk_spread.get("line"),
                        home_cover_prob=spread_prediction.home_cover_prob,
                    )

                spread_response = SpreadPredictionResponse(
                    predicted_spread=round(spread_prediction.predicted_spread, 1),
                    prediction_std=round(spread_prediction.prediction_std, 2),
                    confidence_low=round(spread_prediction.confidence_lower, 1),
                    confidence_high=round(spread_prediction.confidence_upper, 1),
                    home_cover_prob=round(spread_prediction.home_cover_prob, 3),
                    away_cover_prob=round(spread_prediction.away_cover_prob, 3),
                    dk_line=dk_spread.get("line") if dk_spread else None,
                    dk_home_odds=dk_spread.get("home_odds") if dk_spread else None,
                    dk_away_odds=dk_spread.get("away_odds") if dk_spread else None,
                    recommendation=recommendation,
                    edge=round(edge, 4) if edge is not None else None,
                    bet_confidence=bet_confidence,
                )

                logger.info(f"[{game_id}] Spread prediction: {spread_pred}")

        except Exception as e:
            logger.warning(f"[{game_id}] Failed to generate spread prediction: {e}")

    # Fetch DraftKings prop lines for this game
    dk_lines = await _fetch_draftkings_lines(app_state, home_team, away_team)
    logger.info(f"[{game_id}] Got {len(dk_lines)} DraftKings prop lines")

    # Generate player prop predictions
    player_preds = []

    for team, opponent in [(home_team, away_team), (away_team, home_team)]:
        team_players = []
        if app_state.pipeline:
            try:
                team_players = await app_state.pipeline.get_key_players_dynamic(
                    team=team,
                    season=season,
                    positions=["QB", "RB", "WR", "TE"],
                    max_per_position=1,
                    exclude_injured=False,
                )
            except Exception as e:
                logger.debug(f"ESPN roster lookup failed for {team}: {e}")

        if not team_players:
            static_players = KEY_PLAYERS.get(team, [])
            team_players = [
                {"name": p["name"], "position": p["position"], "injury_status": "UNKNOWN"}
                for p in static_players
            ]

        for player_info in team_players:
            player_name = player_info.get("name", "")
            position = player_info.get("position", "")
            injury_status = player_info.get("injury_status", "ACTIVE")
            prop_types = _get_prop_type_for_position(position)

            player_id = await app_state.feature_pipeline.lookup_player_id(
                player_name=player_name,
                season=season,
                position=position,
            )

            if not player_id:
                continue

            for prop_type in prop_types:
                prop_model = value_detector.prop_models.get(prop_type)
                if not prop_model:
                    continue

                try:
                    features = await app_state.feature_pipeline.build_prop_features(
                        game_id=gid,
                        player_id=player_id,
                        player_name=player_name,
                        prop_type=prop_type,
                        season=season,
                        week=week,
                        opponent_team=opponent,
                        position=position,
                    )

                    if features is None or features.features is None:
                        continue

                    prediction = prop_model.predict_player(
                        features=features.features,
                        player_id=player_id,
                        player_name=player_name,
                        game_id=gid,
                        team=team,
                        opponent=opponent,
                        line=None,
                    )

                    confidence = max(0.5, min(0.95, 1.0 - (prediction.prediction_std / prediction.predicted_value) if prediction.predicted_value > 0 else 0.5))

                    # Look up DraftKings line for this player/prop
                    dk_line_info = dk_lines.get((player_name.lower().strip(), prop_type))
                    dk_line = dk_line_info.get("line") if dk_line_info else None
                    dk_over_odds = dk_line_info.get("over_odds") if dk_line_info else None
                    dk_under_odds = dk_line_info.get("under_odds") if dk_line_info else None

                    # Calculate betting recommendation if line available
                    recommendation = None
                    edge = None
                    bet_confidence = None
                    if dk_line is not None:
                        recommendation, edge, bet_confidence = _calculate_betting_edge(
                            predicted_value=prediction.predicted_value,
                            line=dk_line,
                            prediction_std=prediction.prediction_std,
                        )

                    player_preds.append(
                        PropPredictionResponse(
                            player_name=player_name,
                            team=team,
                            opponent=opponent,
                            game_id=gid,
                            prop_type=prop_type,
                            predicted_value=round(prediction.predicted_value, 1),
                            range_low=round(prediction.quantile_25, 1),
                            range_high=round(prediction.quantile_75, 1),
                            confidence=round(confidence, 2),
                            injury_status=injury_status,
                            # DraftKings line info
                            dk_line=dk_line,
                            dk_over_odds=dk_over_odds,
                            dk_under_odds=dk_under_odds,
                            # Betting recommendation
                            recommendation=recommendation,
                            edge=round(edge, 4) if edge is not None else None,
                            bet_confidence=bet_confidence,
                        )
                    )
                except Exception as e:
                    logger.debug(f"Failed to predict {prop_type} for {player_name}: {e}")
                    continue

    player_preds.sort(key=lambda p: (-p.predicted_value, p.prop_type))

    return GamePredictionsResponse(
        game_id=gid,
        home_team=home_team,
        away_team=away_team,
        kickoff=kickoff,
        spread_prediction=spread_pred,
        spread=spread_response,
        player_props=player_preds,
    ).model_dump()


@router.get("/predictions/enhanced")
async def get_enhanced_predictions(
    request: Request,
    game_id: Optional[str] = Query(None, description="Filter to specific game"),
    include_backups: bool = Query(True, description="Include backup player predictions"),
) -> dict[str, Any]:
    """
    Get predictions with full injury context.

    Features:
    - Usage adjustments for teammates of injured players
    - Backup player predictions when starters are Out/Doubtful
    - Widened prediction ranges for uncertain situations
    - Injury summary per game

    Returns enhanced predictions with injury-adjusted values.
    """
    app_state = request.app.state.app_state

    if not app_state.pipeline or not app_state.feature_pipeline:
        return {
            "count": 0,
            "games": [],
            "generated_at": datetime.now().isoformat(),
            "error": "Pipeline not initialized",
        }

    # Get value detector with prop models
    value_detector = app_state.value_detector
    if not value_detector or not value_detector.prop_models:
        return {
            "count": 0,
            "games": [],
            "generated_at": datetime.now().isoformat(),
            "error": "Prop models not loaded",
        }

    # Check if feature pipeline has depth analyzer
    has_depth_analyzer = hasattr(app_state.feature_pipeline, 'depth_analyzer') and app_state.feature_pipeline.depth_analyzer is not None

    # Get upcoming games
    try:
        games_df = await app_state.pipeline.get_upcoming_games()
        if games_df is None or len(games_df) == 0:
            return {
                "count": 0,
                "games": [],
                "generated_at": datetime.now().isoformat(),
            }
        games = games_df.to_dicts()
    except Exception as e:
        logger.error(f"Failed to get upcoming games: {e}")
        return {
            "count": 0,
            "games": [],
            "generated_at": datetime.now().isoformat(),
            "error": str(e),
        }

    # Filter to specific game if requested
    if game_id:
        games = [g for g in games if g.get("game_id") == game_id]

    # Generate predictions for each game
    game_predictions = []

    for game in games:
        gid = game.get("game_id", "")
        home_team = game.get("home_team", "")
        away_team = game.get("away_team", "")
        season = game.get("season", 2024)
        week = game.get("week", 1)
        kickoff = game.get("commence_time") or game.get("gameday") or ""

        if isinstance(kickoff, datetime):
            kickoff = kickoff.isoformat()

        player_preds = []
        backup_preds = []
        injury_summary = {"home_team": {}, "away_team": {}}

        for team, opponent, team_key in [
            (home_team, away_team, "home_team"),
            (away_team, home_team, "away_team"),
        ]:
            # Get injury analysis for team if depth analyzer available
            injured_starters = []
            if has_depth_analyzer:
                try:
                    analysis = await app_state.feature_pipeline.depth_analyzer.analyze_injury_impact(
                        team=team,
                        season=season,
                    )
                    injured_starters = [
                        {"name": ip.player_name, "position": ip.position, "status": ip.injury_status}
                        for ip in analysis.injured_starters
                    ]
                    injury_summary[team_key] = {
                        "injured_starters": injured_starters,
                        "backup_activations": [
                            {"backup": ba.backup_player_name, "replacing": ba.replacing_player_name, "position": ba.position}
                            for ba in analysis.backup_activations
                        ],
                    }
                except Exception as e:
                    logger.debug(f"Failed to get injury analysis for {team}: {e}")

            # Get team players
            team_players = []
            if app_state.pipeline:
                try:
                    team_players = await app_state.pipeline.get_key_players_dynamic(
                        team=team,
                        season=season,
                        positions=["QB", "RB", "WR", "TE"],
                        max_per_position=1,
                        exclude_injured=False,
                    )
                except Exception as e:
                    logger.debug(f"ESPN roster lookup failed for {team}: {e}")

            if not team_players:
                static_players = KEY_PLAYERS.get(team, [])
                team_players = [
                    {"name": p["name"], "position": p["position"], "injury_status": "UNKNOWN"}
                    for p in static_players
                ]

            for player_info in team_players:
                player_name = player_info.get("name", "")
                position = player_info.get("position", "")
                injury_status = player_info.get("injury_status", "ACTIVE")
                prop_types = _get_prop_type_for_position(position)

                player_id = await app_state.feature_pipeline.lookup_player_id(
                    player_name=player_name,
                    season=season,
                    position=position,
                )

                if not player_id:
                    continue

                for prop_type in prop_types:
                    prop_model = value_detector.prop_models.get(prop_type)
                    if not prop_model:
                        continue

                    try:
                        # Use injury-adjusted features if available
                        if has_depth_analyzer:
                            features = await app_state.feature_pipeline.build_injury_adjusted_prop_features(
                                game_id=gid,
                                player_id=player_id,
                                player_name=player_name,
                                prop_type=prop_type,
                                season=season,
                                week=week,
                                opponent_team=opponent,
                                player_team=team,
                                position=position,
                            )

                            uncertainty_mult = features.uncertainty_multiplier
                            is_backup_starter = features.is_backup_starter
                            replacing_player = features.replacing_player
                            usage_boosts = features.usage_boosts_applied
                            feature_dict = features.features
                        else:
                            base_features = await app_state.feature_pipeline.build_prop_features(
                                game_id=gid,
                                player_id=player_id,
                                player_name=player_name,
                                prop_type=prop_type,
                                season=season,
                                week=week,
                                opponent_team=opponent,
                                position=position,
                            )
                            uncertainty_mult = 1.0
                            is_backup_starter = False
                            replacing_player = None
                            usage_boosts = {}
                            feature_dict = base_features.features

                        # Make prediction with uncertainty adjustment
                        prediction = prop_model.predict_player_with_uncertainty(
                            features=feature_dict,
                            player_id=player_id,
                            player_name=player_name,
                            game_id=gid,
                            team=team,
                            opponent=opponent,
                            line=None,
                            uncertainty_multiplier=uncertainty_mult,
                        )

                        # Calculate confidence
                        confidence = max(0.5, min(0.95, 1.0 - (prediction.prediction_std / prediction.predicted_value) if prediction.predicted_value > 0 else 0.5))

                        # Build usage boost reason
                        usage_boost_reason = None
                        if usage_boosts:
                            boost_keys = list(usage_boosts.keys())
                            if injured_starters:
                                injured_names = [s["name"] for s in injured_starters]
                                usage_boost_reason = f"Boost from injured: {', '.join(injured_names[:2])}"

                        player_preds.append(
                            InjuryAdjustedPredictionResponse(
                                player_name=player_name,
                                team=team,
                                opponent=opponent,
                                game_id=gid,
                                prop_type=prop_type,
                                predicted_value=round(prediction.predicted_value, 1),
                                range_low=round(prediction.quantile_25, 1),
                                range_high=round(prediction.quantile_75, 1),
                                confidence=round(confidence, 2),
                                injury_status=injury_status,
                                uncertainty_multiplier=round(uncertainty_mult, 2),
                                is_backup_starter=is_backup_starter,
                                replacing_player=replacing_player,
                                usage_boost_applied=bool(usage_boosts),
                                usage_boost_reason=usage_boost_reason,
                            )
                        )

                    except Exception as e:
                        logger.debug(f"Failed enhanced prediction for {player_name}: {e}")
                        continue

            # Generate backup predictions if requested
            if include_backups and has_depth_analyzer:
                try:
                    backup_features = await app_state.feature_pipeline.get_backup_player_features(
                        team=team,
                        season=season,
                        week=week,
                        opponent_team=opponent,
                        prop_types=["passing_yards", "rushing_yards", "receiving_yards"],
                    )

                    for bf in backup_features:
                        prop_model = value_detector.prop_models.get(bf.prop_type)
                        if not prop_model:
                            continue

                        try:
                            prediction = prop_model.predict_player_with_uncertainty(
                                features=bf.features,
                                player_id=bf.player_id,
                                player_name=bf.player_name,
                                game_id=gid,
                                team=team,
                                opponent=opponent,
                                line=None,
                                uncertainty_multiplier=bf.uncertainty_multiplier,
                            )

                            confidence = max(0.5, min(0.95, 1.0 - (prediction.prediction_std / prediction.predicted_value) if prediction.predicted_value > 0 else 0.5))

                            backup_preds.append(
                                BackupPlayerPrediction(
                                    player_name=bf.player_name,
                                    team=team,
                                    opponent=opponent,
                                    game_id=gid,
                                    prop_type=bf.prop_type,
                                    predicted_value=round(prediction.predicted_value, 1),
                                    range_low=round(prediction.quantile_25, 1),
                                    range_high=round(prediction.quantile_75, 1),
                                    confidence=round(confidence, 2),
                                    injury_status=bf.injury_status,
                                    is_backup=True,
                                    replacing_player=bf.replacing_player or "Unknown",
                                    replacing_injury_status="OUT",
                                    uncertainty_note=f"Elevated uncertainty (x{bf.uncertainty_multiplier:.1f}): backup stepping into starter role",
                                )
                            )

                        except Exception as e:
                            logger.debug(f"Failed backup prediction for {bf.player_name}: {e}")

                except Exception as e:
                    logger.debug(f"Failed to get backup features for {team}: {e}")

        # Sort predictions
        player_preds.sort(key=lambda p: (-p.predicted_value, p.prop_type))
        backup_preds.sort(key=lambda p: (-p.predicted_value, p.prop_type))

        game_predictions.append(
            EnhancedGamePredictionsResponse(
                game_id=gid,
                home_team=home_team,
                away_team=away_team,
                kickoff=kickoff,
                spread_prediction=None,
                player_props=player_preds,
                backup_player_props=backup_preds,
                injury_summary=injury_summary,
            )
        )

    return {
        "count": len(game_predictions),
        "games": game_predictions,
        "generated_at": datetime.now().isoformat(),
        "includes_injury_adjustments": has_depth_analyzer,
    }
