"""
Injury impact feature engineering.

Computes health-related features from SIC Score data:
- Team health scores
- Position group health
- Key player injury impact
- Health advantage differential
"""
from datetime import datetime
from typing import Optional

import polars as pl
from loguru import logger

from .base import BaseFeatureBuilder, FeatureSet


class InjuryFeatureBuilder(BaseFeatureBuilder):
    """
    Builds injury impact features from SIC Score data.

    Features capture the expected performance impact of injuries
    by considering player value, injury type, and position importance.

    Key feature categories:
    - Overall team health score (0-100)
    - Position group health (QB, skill, OL, DL, secondary)
    - Key player availability
    - Health advantage differential
    """

    # Position group weights for overall health calculation
    POSITION_WEIGHTS = {
        "qb": 2.0,  # QB most important
        "skill": 1.5,  # RB, WR, TE
        "oline": 1.5,  # Offensive line
        "dline": 1.0,  # Defensive line
        "linebacker": 1.0,
        "secondary": 1.2,  # CBs crucial
        "special_teams": 0.3,
    }

    HEALTH_FEATURES = [
        "team_health_score",
        "qb_health",
        "skill_health",
        "oline_health",
        "dline_health",
        "linebacker_health",
        "secondary_health",
        "offensive_health",
        "defensive_health",
    ]

    MATCHUP_FEATURES = [
        "home_team_health",
        "away_team_health",
        "health_advantage",
        "home_offense_health",
        "away_offense_health",
        "home_defense_health",
        "away_defense_health",
        "offense_vs_defense_health",
    ]

    def __init__(
        self,
        sic_client=None,
        cache=None,
        cache_ttl_seconds: int = 21600,  # 6 hours
    ):
        super().__init__(cache=cache, cache_ttl_seconds=cache_ttl_seconds)
        self.sic_client = sic_client

    def get_feature_names(self) -> list[str]:
        """Get all feature names this builder produces."""
        return self.HEALTH_FEATURES + self.MATCHUP_FEATURES

    async def build_features(
        self,
        team: str,
        health_data: Optional[dict] = None,
    ) -> FeatureSet:
        """
        Build injury features for a team.

        Args:
            team: Team abbreviation
            health_data: Optional pre-fetched health data from SIC

        Returns:
            FeatureSet with injury features
        """
        # Check cache first
        cache_key = self._get_cache_key("injury", team)
        cached = await self._get_cached(cache_key)
        if cached is not None:
            return cached

        self.logger.debug(f"Building injury features: {team}")

        features = {}

        if health_data:
            # Use provided health data
            features = self._extract_health_features(health_data)
        elif self.sic_client:
            # Fetch from SIC Score
            try:
                health_data = await self.sic_client.get_team_health(team)
                features = self._extract_health_features(health_data)
            except Exception as e:
                self.logger.warning(f"Failed to get SIC data for {team}: {e}")
                features = self._get_default_health_features()
        else:
            # No SIC client - use defaults
            features = self._get_default_health_features()

        # Create feature set
        feature_set = FeatureSet(
            features=features,
            team=team,
        )

        # Cache the result
        await self._set_cached(cache_key, feature_set)

        return feature_set

    def _extract_health_features(
        self,
        health_data: dict,
    ) -> dict[str, float]:
        """Extract features from SIC Score health data."""
        features = {}

        # Overall team health
        features["team_health_score"] = health_data.get("overall_health", 100.0)

        # Position group health
        position_groups = health_data.get("position_groups", {})
        features["qb_health"] = position_groups.get("qb", 100.0)
        features["skill_health"] = position_groups.get("skill", 100.0)
        features["oline_health"] = position_groups.get("oline", 100.0)
        features["dline_health"] = position_groups.get("dline", 100.0)
        features["linebacker_health"] = position_groups.get("linebacker", 100.0)
        features["secondary_health"] = position_groups.get("secondary", 100.0)

        # Aggregate offensive/defensive health
        features["offensive_health"] = (
            features["qb_health"] * 0.4
            + features["skill_health"] * 0.35
            + features["oline_health"] * 0.25
        )

        features["defensive_health"] = (
            features["dline_health"] * 0.35
            + features["linebacker_health"] * 0.30
            + features["secondary_health"] * 0.35
        )

        return features

    def _get_default_health_features(self) -> dict[str, float]:
        """Get default health features (fully healthy)."""
        return {
            "team_health_score": 100.0,
            "qb_health": 100.0,
            "skill_health": 100.0,
            "oline_health": 100.0,
            "dline_health": 100.0,
            "linebacker_health": 100.0,
            "secondary_health": 100.0,
            "offensive_health": 100.0,
            "defensive_health": 100.0,
        }

    async def build_matchup_features(
        self,
        home_team: str,
        away_team: str,
        home_health: Optional[dict] = None,
        away_health: Optional[dict] = None,
    ) -> FeatureSet:
        """
        Build injury features for a matchup.

        Args:
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            home_health: Optional pre-fetched home team health
            away_health: Optional pre-fetched away team health

        Returns:
            FeatureSet with matchup injury features
        """
        cache_key = self._get_cache_key("injury_matchup", home_team, away_team)
        cached = await self._get_cached(cache_key)
        if cached is not None:
            return cached

        # Get individual team features
        home_features = await self.build_features(home_team, home_health)
        away_features = await self.build_features(away_team, away_health)

        # Build matchup features
        features = {}

        # Overall health
        features["home_team_health"] = home_features.features.get("team_health_score", 100.0)
        features["away_team_health"] = away_features.features.get("team_health_score", 100.0)
        features["health_advantage"] = (
            features["home_team_health"] - features["away_team_health"]
        )

        # Offensive health
        features["home_offense_health"] = home_features.features.get("offensive_health", 100.0)
        features["away_offense_health"] = away_features.features.get("offensive_health", 100.0)

        # Defensive health
        features["home_defense_health"] = home_features.features.get("defensive_health", 100.0)
        features["away_defense_health"] = away_features.features.get("defensive_health", 100.0)

        # Matchup advantage (home offense vs away defense, away offense vs home defense)
        # Positive means home team advantage
        home_off_vs_away_def = (
            features["home_offense_health"] - features["away_defense_health"]
        )
        away_off_vs_home_def = (
            features["away_offense_health"] - features["home_defense_health"]
        )
        features["offense_vs_defense_health"] = home_off_vs_away_def - away_off_vs_home_def

        # Add position group differentials
        for pos in ["qb", "skill", "oline", "dline", "linebacker", "secondary"]:
            home_val = home_features.features.get(f"{pos}_health", 100.0)
            away_val = away_features.features.get(f"{pos}_health", 100.0)
            features[f"{pos}_health_diff"] = home_val - away_val

        # Create feature set
        feature_set = FeatureSet(
            features=features,
            game_id=f"{home_team}_vs_{away_team}",
        )

        await self._set_cached(cache_key, feature_set)
        return feature_set

    async def build_player_injury_features(
        self,
        player_id: str,
        team: str,
        position: str,
        injury_data: Optional[dict] = None,
    ) -> FeatureSet:
        """
        Build injury features for a specific player.

        Args:
            player_id: nflverse player ID
            team: Player's team
            position: Player position
            injury_data: Optional injury info for the player

        Returns:
            FeatureSet with player injury features
        """
        features = {}

        if injury_data:
            # Player has injury data
            features["player_is_injured"] = 1.0
            features["player_injury_status"] = self._status_to_score(
                injury_data.get("status", "")
            )
            features["player_injury_impact"] = injury_data.get("impact_score", 0.0)
            features["games_missed"] = injury_data.get("games_missed", 0)
        else:
            # Player is healthy
            features["player_is_injured"] = 0.0
            features["player_injury_status"] = 1.0  # Fully healthy
            features["player_injury_impact"] = 0.0
            features["games_missed"] = 0

        # Get team health for context
        team_features = await self.build_features(team)
        position_group = self._position_to_group(position)
        features["position_group_health"] = team_features.features.get(
            f"{position_group}_health", 100.0
        )

        return FeatureSet(
            features=features,
            player_id=player_id,
            team=team,
        )

    def _status_to_score(self, status: str) -> float:
        """Convert injury status to availability score (0-1)."""
        status_lower = status.lower()
        if status_lower in ["out", "ir", "pup", "nfi"]:
            return 0.0
        elif status_lower == "doubtful":
            return 0.15
        elif status_lower == "questionable":
            return 0.50
        elif status_lower == "probable":
            return 0.85
        else:
            return 1.0  # Healthy

    def _position_to_group(self, position: str) -> str:
        """Map position to position group."""
        position_upper = position.upper()

        if position_upper == "QB":
            return "qb"
        elif position_upper in ["RB", "FB", "WR", "TE"]:
            return "skill"
        elif position_upper in ["T", "G", "C", "OL", "OT", "OG"]:
            return "oline"
        elif position_upper in ["DE", "DT", "NT", "DL", "EDGE"]:
            return "dline"
        elif position_upper in ["LB", "ILB", "OLB", "MLB"]:
            return "linebacker"
        elif position_upper in ["CB", "S", "FS", "SS", "DB"]:
            return "secondary"
        elif position_upper in ["K", "P", "LS"]:
            return "special_teams"
        else:
            return "skill"  # Default


def calculate_injury_adjusted_projection(
    base_projection: float,
    player_health_score: float,
    position_group_health: float,
) -> float:
    """
    Adjust a player projection based on injury context.

    Args:
        base_projection: Base statistical projection
        player_health_score: Player's own health (0-1)
        position_group_health: Team's position group health (0-100)

    Returns:
        Adjusted projection
    """
    # Player availability adjustment
    player_factor = player_health_score

    # Position group adjustment (slight impact from supporting cast)
    group_factor = 0.8 + (0.2 * position_group_health / 100)

    return base_projection * player_factor * group_factor


def estimate_injury_variance(
    status: str,
    injury_history: int = 0,
) -> float:
    """
    Estimate projection variance based on injury status.

    Players with injury concerns have higher projection variance.

    Args:
        status: Current injury status
        injury_history: Number of injuries in past 2 seasons

    Returns:
        Variance multiplier (1.0 = normal, >1.0 = higher variance)
    """
    status_lower = status.lower()

    # Base variance by status
    if status_lower in ["out", "ir"]:
        return 0.0  # Not playing
    elif status_lower == "doubtful":
        return 2.5
    elif status_lower == "questionable":
        return 1.8
    elif status_lower == "probable":
        return 1.2
    else:
        base_variance = 1.0

    # Adjust for injury history
    history_factor = 1.0 + (0.1 * min(injury_history, 5))

    return base_variance * history_factor
