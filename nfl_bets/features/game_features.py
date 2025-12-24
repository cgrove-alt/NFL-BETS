"""
Game context feature engineering.

Computes situational and contextual features:
- Rest days and schedule advantages
- Travel distance and timezone changes
- Weather conditions (outdoor games)
- Vegas lines and implied totals
- Rivalry and divisional matchups
"""
import math
from datetime import datetime, timedelta
from typing import Optional

import polars as pl
from loguru import logger

from .base import BaseFeatureBuilder, FeatureMetadata, FeatureSet


# NFL team stadium locations (lat, lon) for travel calculations
TEAM_LOCATIONS = {
    "ARI": (33.5276, -112.2626),  # State Farm Stadium, Glendale
    "ATL": (33.7553, -84.4006),  # Mercedes-Benz Stadium, Atlanta
    "BAL": (39.2780, -76.6227),  # M&T Bank Stadium, Baltimore
    "BUF": (42.7738, -78.7870),  # Highmark Stadium, Orchard Park
    "CAR": (35.2258, -80.8528),  # Bank of America Stadium, Charlotte
    "CHI": (41.8623, -87.6167),  # Soldier Field, Chicago
    "CIN": (39.0954, -84.5160),  # Paycor Stadium, Cincinnati
    "CLE": (41.5061, -81.6995),  # Cleveland Browns Stadium
    "DAL": (32.7473, -97.0945),  # AT&T Stadium, Arlington
    "DEN": (39.7439, -105.0201),  # Empower Field, Denver
    "DET": (42.3400, -83.0456),  # Ford Field, Detroit
    "GB": (44.5013, -88.0622),  # Lambeau Field, Green Bay
    "HOU": (29.6847, -95.4107),  # NRG Stadium, Houston
    "IND": (39.7601, -86.1639),  # Lucas Oil Stadium, Indianapolis
    "JAX": (30.3239, -81.6373),  # TIAA Bank Field, Jacksonville
    "KC": (39.0489, -94.4839),  # Arrowhead Stadium, Kansas City
    "LAC": (33.9535, -118.3392),  # SoFi Stadium, Inglewood
    "LAR": (33.9535, -118.3392),  # SoFi Stadium, Inglewood
    "LV": (36.0909, -115.1833),  # Allegiant Stadium, Las Vegas
    "MIA": (25.9580, -80.2389),  # Hard Rock Stadium, Miami Gardens
    "MIN": (44.9736, -93.2575),  # U.S. Bank Stadium, Minneapolis
    "NE": (42.0909, -71.2643),  # Gillette Stadium, Foxborough
    "NO": (29.9511, -90.0812),  # Caesars Superdome, New Orleans
    "NYG": (40.8128, -74.0742),  # MetLife Stadium, East Rutherford
    "NYJ": (40.8128, -74.0742),  # MetLife Stadium, East Rutherford
    "PHI": (39.9008, -75.1675),  # Lincoln Financial Field, Philadelphia
    "PIT": (40.4468, -80.0158),  # Acrisure Stadium, Pittsburgh
    "SEA": (47.5952, -122.3316),  # Lumen Field, Seattle
    "SF": (37.4033, -121.9694),  # Levi's Stadium, Santa Clara
    "TB": (27.9759, -82.5033),  # Raymond James Stadium, Tampa
    "TEN": (36.1665, -86.7713),  # Nissan Stadium, Nashville
    "WAS": (38.9076, -76.8645),  # FedExField, Landover
}

# Timezone offsets (hours from UTC during regular season)
TEAM_TIMEZONES = {
    "ARI": -7,  # MST (no DST)
    "ATL": -5,
    "BAL": -5,
    "BUF": -5,
    "CAR": -5,
    "CHI": -6,
    "CIN": -5,
    "CLE": -5,
    "DAL": -6,
    "DEN": -7,
    "DET": -5,
    "GB": -6,
    "HOU": -6,
    "IND": -5,
    "JAX": -5,
    "KC": -6,
    "LAC": -8,
    "LAR": -8,
    "LV": -8,
    "MIA": -5,
    "MIN": -6,
    "NE": -5,
    "NO": -6,
    "NYG": -5,
    "NYJ": -5,
    "PHI": -5,
    "PIT": -5,
    "SEA": -8,
    "SF": -8,
    "TB": -5,
    "TEN": -6,
    "WAS": -5,
}

# Indoor stadiums (dome or retractable roof)
INDOOR_STADIUMS = {
    "ARI",  # Retractable roof
    "ATL",  # Retractable roof
    "DAL",  # Retractable roof
    "DET",  # Dome
    "HOU",  # Retractable roof
    "IND",  # Retractable roof
    "LAC",  # Indoor
    "LAR",  # Indoor
    "LV",  # Indoor
    "MIN",  # Indoor
    "NO",  # Dome
}

# Division mappings for rivalry detection
AFC_EAST = {"BUF", "MIA", "NE", "NYJ"}
AFC_NORTH = {"BAL", "CIN", "CLE", "PIT"}
AFC_SOUTH = {"HOU", "IND", "JAX", "TEN"}
AFC_WEST = {"DEN", "KC", "LAC", "LV"}
NFC_EAST = {"DAL", "NYG", "PHI", "WAS"}
NFC_NORTH = {"CHI", "DET", "GB", "MIN"}
NFC_SOUTH = {"ATL", "CAR", "NO", "TB"}
NFC_WEST = {"ARI", "LAR", "SEA", "SF"}

DIVISIONS = {
    "AFC_EAST": AFC_EAST,
    "AFC_NORTH": AFC_NORTH,
    "AFC_SOUTH": AFC_SOUTH,
    "AFC_WEST": AFC_WEST,
    "NFC_EAST": NFC_EAST,
    "NFC_NORTH": NFC_NORTH,
    "NFC_SOUTH": NFC_SOUTH,
    "NFC_WEST": NFC_WEST,
}

AFC_TEAMS = AFC_EAST | AFC_NORTH | AFC_SOUTH | AFC_WEST
NFC_TEAMS = NFC_EAST | NFC_NORTH | NFC_SOUTH | NFC_WEST


class GameFeatureBuilder(BaseFeatureBuilder):
    """
    Builds game-context features for spread and prop predictions.

    Key feature categories:
    - Rest: Days since last game for each team
    - Travel: Distance traveled, timezone changes
    - Weather: Temperature, wind, precipitation (outdoor only)
    - Vegas: Opening line, current line, implied total
    - Rivalry: Divisional, conference, historical matchup
    """

    CONTEXT_FEATURES = [
        "home_rest_days",
        "away_rest_days",
        "rest_advantage",
        "home_travel_distance",
        "away_travel_distance",
        "travel_advantage",
        "home_timezone_change",
        "away_timezone_change",
        "is_divisional",
        "is_conference",
        "is_primetime",
        "is_outdoor",
        "opening_spread",
        "current_spread",
        "line_movement",
        "implied_total",
        "home_implied_score",
        "away_implied_score",
    ]

    WEATHER_FEATURES = [
        "temperature",
        "wind_speed",
        "precipitation_prob",
        "is_cold_game",
        "is_windy",
    ]

    def __init__(
        self,
        cache=None,
        cache_ttl_seconds: int = 21600,
    ):
        super().__init__(cache=cache, cache_ttl_seconds=cache_ttl_seconds)

    def get_feature_names(self) -> list[str]:
        """Get all feature names this builder produces."""
        return self.CONTEXT_FEATURES + self.WEATHER_FEATURES

    async def build_features(
        self,
        schedules_df: pl.DataFrame,
        game_id: str,
        home_team: str,
        away_team: str,
        odds_data: Optional[dict] = None,
        weather_data: Optional[dict] = None,
    ) -> FeatureSet:
        """
        Build all game context features.

        Args:
            schedules_df: Schedules DataFrame with game history
            game_id: Current game ID
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            odds_data: Optional odds data from Odds API
            weather_data: Optional weather data

        Returns:
            FeatureSet with computed game features
        """
        # Check cache first
        cache_key = self._get_cache_key("game", game_id)
        cached = await self._get_cached(cache_key)
        if cached is not None:
            return cached

        self.logger.debug(f"Building game features: {game_id}")

        features = {}

        # Build rest features
        rest_features = self._build_rest_features(schedules_df, game_id, home_team, away_team)
        features.update(rest_features)

        # Build travel features
        travel_features = self._build_travel_features(home_team, away_team)
        features.update(travel_features)

        # Build rivalry features
        rivalry_features = self._build_rivalry_features(home_team, away_team)
        features.update(rivalry_features)

        # Build venue features
        venue_features = self._build_venue_features(home_team)
        features.update(venue_features)

        # Build Vegas features
        if odds_data:
            vegas_features = self._build_vegas_features(odds_data)
            features.update(vegas_features)
        else:
            features.update(self._get_default_vegas_features())

        # Build weather features
        if weather_data:
            weather_features = self._build_weather_features(weather_data)
            features.update(weather_features)
        else:
            features.update(self._get_default_weather_features(home_team))

        # Create feature set
        feature_set = FeatureSet(
            features=features,
            game_id=game_id,
        )

        # Cache the result
        await self._set_cached(cache_key, feature_set)

        return feature_set

    def _build_rest_features(
        self,
        schedules_df: pl.DataFrame,
        game_id: str,
        home_team: str,
        away_team: str,
    ) -> dict[str, float]:
        """Build rest advantage features."""
        features = {
            "home_rest_days": 7.0,  # Default to standard week
            "away_rest_days": 7.0,
            "rest_advantage": 0.0,
        }

        if len(schedules_df) == 0:
            return features

        # Get current game info
        current_game = schedules_df.filter(pl.col("game_id") == game_id)
        if len(current_game) == 0:
            return features

        current_date = current_game["game_date"][0]
        if current_date is None:
            return features

        # Find home team's previous game
        home_prev = (
            schedules_df.filter(
                ((pl.col("home_team") == home_team) | (pl.col("away_team") == home_team))
                & (pl.col("game_date") < current_date)
            )
            .sort("game_date", descending=True)
            .head(1)
        )

        if len(home_prev) > 0:
            prev_date = home_prev["game_date"][0]
            if prev_date is not None:
                if isinstance(current_date, datetime) and isinstance(prev_date, datetime):
                    features["home_rest_days"] = (current_date - prev_date).days
                else:
                    features["home_rest_days"] = 7.0

        # Find away team's previous game
        away_prev = (
            schedules_df.filter(
                ((pl.col("home_team") == away_team) | (pl.col("away_team") == away_team))
                & (pl.col("game_date") < current_date)
            )
            .sort("game_date", descending=True)
            .head(1)
        )

        if len(away_prev) > 0:
            prev_date = away_prev["game_date"][0]
            if prev_date is not None:
                if isinstance(current_date, datetime) and isinstance(prev_date, datetime):
                    features["away_rest_days"] = (current_date - prev_date).days
                else:
                    features["away_rest_days"] = 7.0

        # Calculate rest advantage (positive = home advantage)
        features["rest_advantage"] = features["home_rest_days"] - features["away_rest_days"]

        return features

    def _build_travel_features(
        self,
        home_team: str,
        away_team: str,
    ) -> dict[str, float]:
        """Build travel distance and timezone features."""
        features = {
            "home_travel_distance": 0.0,  # Home team doesn't travel
            "away_travel_distance": 0.0,
            "travel_advantage": 0.0,
            "home_timezone_change": 0,
            "away_timezone_change": 0,
        }

        # Get locations
        home_loc = TEAM_LOCATIONS.get(home_team)
        away_loc = TEAM_LOCATIONS.get(away_team)

        if home_loc and away_loc:
            # Calculate distance (Haversine formula)
            distance = self._haversine_distance(away_loc, home_loc)
            features["away_travel_distance"] = distance
            features["travel_advantage"] = distance  # Home advantage

        # Calculate timezone changes
        home_tz = TEAM_TIMEZONES.get(home_team, -6)
        away_tz = TEAM_TIMEZONES.get(away_team, -6)
        features["away_timezone_change"] = abs(home_tz - away_tz)

        return features

    def _haversine_distance(
        self,
        point1: tuple[float, float],
        point2: tuple[float, float],
    ) -> float:
        """
        Calculate distance between two lat/lon points in miles.

        Uses the Haversine formula.
        """
        lat1, lon1 = point1
        lat2, lon2 = point2

        # Convert to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.asin(math.sqrt(a))

        # Earth's radius in miles
        r = 3959

        return c * r

    def _build_rivalry_features(
        self,
        home_team: str,
        away_team: str,
    ) -> dict[str, float]:
        """Build rivalry and divisional features."""
        features = {
            "is_divisional": 0.0,
            "is_conference": 0.0,
        }

        # Check if divisional game
        for division_teams in DIVISIONS.values():
            if home_team in division_teams and away_team in division_teams:
                features["is_divisional"] = 1.0
                features["is_conference"] = 1.0
                break

        # Check if conference game (not divisional)
        if features["is_divisional"] == 0.0:
            home_is_afc = home_team in AFC_TEAMS
            away_is_afc = away_team in AFC_TEAMS
            if home_is_afc == away_is_afc:
                features["is_conference"] = 1.0

        return features

    def _build_venue_features(
        self,
        home_team: str,
    ) -> dict[str, float]:
        """Build venue-related features."""
        return {
            "is_outdoor": 0.0 if home_team in INDOOR_STADIUMS else 1.0,
            "is_primetime": 0.0,  # Would need game time to determine
        }

    def _build_vegas_features(
        self,
        odds_data: dict,
    ) -> dict[str, float]:
        """Build Vegas line features from odds data."""
        features = {
            "opening_spread": 0.0,
            "current_spread": 0.0,
            "line_movement": 0.0,
            "implied_total": 45.0,  # Default
            "home_implied_score": 22.5,
            "away_implied_score": 22.5,
        }

        # Extract spread
        if "spreads" in odds_data:
            for outcome in odds_data.get("spreads", {}).get("outcomes", []):
                if outcome.get("name") == odds_data.get("home_team"):
                    features["current_spread"] = outcome.get("point", 0.0)
                    break

        # Extract total
        if "totals" in odds_data:
            for outcome in odds_data.get("totals", {}).get("outcomes", []):
                if outcome.get("name") == "Over":
                    features["implied_total"] = outcome.get("point", 45.0)
                    break

        # Calculate implied scores
        spread = features["current_spread"]
        total = features["implied_total"]
        features["home_implied_score"] = (total - spread) / 2
        features["away_implied_score"] = (total + spread) / 2

        # Line movement (would need historical data)
        if "opening_spread" in odds_data:
            features["opening_spread"] = odds_data["opening_spread"]
            features["line_movement"] = (
                features["current_spread"] - features["opening_spread"]
            )

        return features

    def _get_default_vegas_features(self) -> dict[str, float]:
        """Get default Vegas features when no odds data available."""
        return {
            "opening_spread": 0.0,
            "current_spread": 0.0,
            "line_movement": 0.0,
            "implied_total": 45.0,
            "home_implied_score": 22.5,
            "away_implied_score": 22.5,
        }

    def _build_weather_features(
        self,
        weather_data: dict,
    ) -> dict[str, float]:
        """Build weather features from weather data."""
        features = {
            "temperature": weather_data.get("temperature", 65.0),
            "wind_speed": weather_data.get("wind_speed", 5.0),
            "precipitation_prob": weather_data.get("precipitation_prob", 0.0),
            "is_cold_game": 1.0 if weather_data.get("temperature", 65) < 40 else 0.0,
            "is_windy": 1.0 if weather_data.get("wind_speed", 5) > 15 else 0.0,
        }
        return features

    def _get_default_weather_features(
        self,
        home_team: str,
    ) -> dict[str, float]:
        """Get default weather features (indoor or mild conditions)."""
        is_indoor = home_team in INDOOR_STADIUMS

        return {
            "temperature": 72.0 if is_indoor else 55.0,
            "wind_speed": 0.0 if is_indoor else 8.0,
            "precipitation_prob": 0.0,
            "is_cold_game": 0.0,
            "is_windy": 0.0,
        }


def get_division(team: str) -> Optional[str]:
    """Get the division for a team."""
    for div_name, teams in DIVISIONS.items():
        if team in teams:
            return div_name
    return None


def get_conference(team: str) -> Optional[str]:
    """Get the conference for a team."""
    if team in AFC_TEAMS:
        return "AFC"
    elif team in NFC_TEAMS:
        return "NFC"
    return None


def is_primetime_game(
    game_time: datetime,
    day_of_week: int,
) -> bool:
    """
    Determine if a game is primetime.

    Primetime games:
    - Sunday Night Football (Sunday 8:20 PM ET)
    - Monday Night Football (Monday 8:15 PM ET)
    - Thursday Night Football (Thursday 8:20 PM ET)
    """
    hour = game_time.hour

    # Sunday night
    if day_of_week == 6 and hour >= 20:
        return True

    # Monday night
    if day_of_week == 0 and hour >= 20:
        return True

    # Thursday night
    if day_of_week == 3 and hour >= 20:
        return True

    return False
