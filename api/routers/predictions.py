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


class GamePredictionsResponse(BaseModel):
    """Response model for predictions for a single game."""

    game_id: str
    home_team: str
    away_team: str
    kickoff: str
    spread_prediction: Optional[float] = None
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

    for game in games:
        gid = game.get("game_id", "")
        home_team = game.get("home_team", "")
        away_team = game.get("away_team", "")
        season = game.get("season", 2024)
        week = game.get("week", 1)
        kickoff = game.get("commence_time") or game.get("gameday") or ""

        if isinstance(kickoff, datetime):
            kickoff = kickoff.isoformat()

        # Get spread prediction if available
        spread_pred = None
        # TODO: Add spread model prediction here

        # Get player predictions for both teams
        player_preds = []

        for team, opponent in [(home_team, away_team), (away_team, home_team)]:
            team_players = KEY_PLAYERS.get(team, [])

            for player_info in team_players:
                player_name = player_info["name"]
                position = player_info["position"]
                prop_types = _get_prop_type_for_position(position)

                for prop_type in prop_types:
                    # Check if we have this prop model
                    prop_model = value_detector.prop_models.get(prop_type)
                    if not prop_model:
                        continue

                    try:
                        # Build features for this player
                        features = await app_state.feature_pipeline.build_prop_features(
                            game_id=gid,
                            player_id=player_name,
                            player_name=player_name,
                            prop_type=prop_type,
                            season=season,
                            week=week,
                            opponent_team=opponent,
                        )

                        if features is None or features.features is None:
                            continue

                        # Make prediction
                        prediction = prop_model.predict_player(
                            features=features.features,
                            player_id=player_name,
                            player_name=player_name,
                            game_id=gid,
                            team=team,
                            opponent=opponent,
                            line=None,  # No line - raw prediction
                        )

                        # Calculate confidence from std dev
                        confidence = max(0.5, min(0.95, 1.0 - (prediction.prediction_std / prediction.predicted_value) if prediction.predicted_value > 0 else 0.5))

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
                player_props=player_preds,
            )
        )

    return {
        "count": len(game_predictions),
        "games": game_predictions,
        "generated_at": datetime.now().isoformat(),
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
    result = await get_all_predictions(request, game_id=game_id)

    if result.get("games"):
        return result["games"][0]

    return {"error": "Game not found", "game_id": game_id}
