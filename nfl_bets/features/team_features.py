"""
Team-level feature engineering.

Computes rolling team performance metrics from play-by-play data:
- EPA (Expected Points Added) metrics
- Passing efficiency (CPOE, air yards)
- Rushing efficiency
- Defensive performance
- Situational efficiency (red zone, third down)
"""
from datetime import datetime
from typing import Optional

import polars as pl
from loguru import logger

from .base import (
    BaseFeatureBuilder,
    FeatureMetadata,
    FeatureSet,
    RollingWindowMixin,
)


class TeamFeatureBuilder(BaseFeatureBuilder, RollingWindowMixin):
    """
    Builds team-level features from play-by-play data.

    Features are computed as rolling averages over recent games
    to capture current team form while avoiding look-ahead bias.

    Key feature categories:
    - EPA metrics: Overall, passing, and rushing efficiency
    - Success rate: % of plays with positive EPA
    - Explosive plays: Big play rate
    - CPOE: Completion percentage over expected
    - Situational: Red zone, third down, late game
    - Defensive: EPA allowed, pressure rate
    """

    # Feature definitions
    OFFENSIVE_FEATURES = [
        "epa_per_play",
        "epa_per_pass",
        "epa_per_rush",
        "success_rate",
        "explosive_play_rate",
        "negative_play_rate",
    ]

    PASSING_FEATURES = [
        "cpoe",
        "air_yards_per_attempt",
        "yac_per_completion",
        "sack_rate",
        "interception_rate",
    ]

    RUSHING_FEATURES = [
        "yards_per_carry",
        "rush_success_rate",
        "stuffed_rate",
    ]

    SITUATIONAL_FEATURES = [
        "red_zone_td_rate",
        "third_down_conv_rate",
        "first_down_rate",
    ]

    DEFENSIVE_FEATURES = [
        "epa_allowed_per_play",
        "epa_allowed_per_pass",
        "epa_allowed_per_rush",
        "defensive_success_rate",
    ]

    def __init__(
        self,
        windows: list[int] | None = None,
        cache=None,
        cache_ttl_seconds: int = 21600,
    ):
        super().__init__(cache=cache, cache_ttl_seconds=cache_ttl_seconds)
        self.windows = windows or self.DEFAULT_WINDOWS

    def get_feature_names(self) -> list[str]:
        """Get all feature names this builder produces."""
        names = []
        all_metrics = (
            self.OFFENSIVE_FEATURES
            + self.PASSING_FEATURES
            + self.RUSHING_FEATURES
            + self.SITUATIONAL_FEATURES
            + self.DEFENSIVE_FEATURES
        )

        for metric in all_metrics:
            for window in self.windows:
                names.append(f"{metric}_{self._window_label(window)}")

        return names

    async def build_features(
        self,
        pbp_df: pl.DataFrame,
        team: str,
        season: int,
        week: int,
        schedules_df: Optional[pl.DataFrame] = None,
    ) -> FeatureSet:
        """
        Build all team features for a specific game.

        Args:
            pbp_df: Play-by-play DataFrame with EPA columns
            team: Team abbreviation
            season: NFL season year
            week: Week number
            schedules_df: Optional schedules for game ordering

        Returns:
            FeatureSet with computed team features
        """
        # Check cache first
        cache_key = self._get_cache_key("team", team, season, week)
        cached = await self._get_cached(cache_key)
        if cached is not None:
            return cached

        self.logger.debug(f"Building team features: {team} S{season} W{week}")

        # Filter to team's offensive plays before this week
        team_pbp = pbp_df.filter(
            (pl.col("posteam") == team)
            & (pl.col("season") == season)
            & (pl.col("week") < week)
            & (pl.col("play_type").is_in(["pass", "run"]))
        )

        # Filter defensive plays (team on defense)
        def_pbp = pbp_df.filter(
            (pl.col("defteam") == team)
            & (pl.col("season") == season)
            & (pl.col("week") < week)
            & (pl.col("play_type").is_in(["pass", "run"]))
        )

        features = {}
        metadata = {}

        # Build offensive features
        off_features = self._build_offensive_features(team_pbp, team)
        features.update(off_features)

        # Build passing features
        pass_features = self._build_passing_features(team_pbp, team)
        features.update(pass_features)

        # Build rushing features
        rush_features = self._build_rushing_features(team_pbp, team)
        features.update(rush_features)

        # Build situational features
        sit_features = self._build_situational_features(team_pbp, team)
        features.update(sit_features)

        # Build defensive features
        def_features = self._build_defensive_features(def_pbp, team)
        features.update(def_features)

        # Create feature set
        feature_set = FeatureSet(
            features=features,
            metadata=metadata,
            game_id=f"{season}_{week}_{team}",
            team=team,
        )

        # Cache the result
        await self._set_cached(cache_key, feature_set)

        return feature_set

    def _build_offensive_features(
        self,
        pbp_df: pl.DataFrame,
        team: str,
    ) -> dict[str, float]:
        """Build offensive EPA and efficiency features."""
        if len(pbp_df) == 0:
            return self._get_default_offensive_features()

        # Compute game-level aggregates
        game_stats = (
            pbp_df.group_by("game_id")
            .agg(
                [
                    pl.col("epa").mean().alias("epa_per_play"),
                    pl.col("epa")
                    .filter(pl.col("play_type") == "pass")
                    .mean()
                    .alias("epa_per_pass"),
                    pl.col("epa")
                    .filter(pl.col("play_type") == "run")
                    .mean()
                    .alias("epa_per_rush"),
                    (pl.col("epa") > 0).mean().alias("success_rate"),
                    (
                        (pl.col("yards_gained") >= 20) | (pl.col("touchdown") == 1)
                    )
                    .mean()
                    .alias("explosive_play_rate"),
                    (pl.col("epa") < -0.5).mean().alias("negative_play_rate"),
                ]
            )
            .sort("game_id")
        )

        if len(game_stats) == 0:
            return self._get_default_offensive_features()

        # Compute rolling features
        features = {}
        metrics = [
            "epa_per_play",
            "epa_per_pass",
            "epa_per_rush",
            "success_rate",
            "explosive_play_rate",
            "negative_play_rate",
        ]

        for metric in metrics:
            values = game_stats[metric].to_list()
            for window in self.windows:
                label = self._window_label(window)
                key = f"{metric}_{label}"
                features[key] = self._handle_missing(
                    self._calculate_rolling_mean(values, window)
                )

        return features

    def _get_default_offensive_features(self) -> dict[str, float]:
        """Get default values for offensive features."""
        defaults = {}
        metrics = [
            "epa_per_play",
            "epa_per_pass",
            "epa_per_rush",
            "success_rate",
            "explosive_play_rate",
            "negative_play_rate",
        ]

        for metric in metrics:
            for window in self.windows:
                label = self._window_label(window)
                defaults[f"{metric}_{label}"] = 0.0

        return defaults

    def _build_passing_features(
        self,
        pbp_df: pl.DataFrame,
        team: str,
    ) -> dict[str, float]:
        """Build passing efficiency features."""
        pass_plays = pbp_df.filter(pl.col("play_type") == "pass")

        if len(pass_plays) == 0:
            return self._get_default_passing_features()

        # Compute game-level passing stats
        game_stats = (
            pass_plays.group_by("game_id")
            .agg(
                [
                    pl.col("cpoe").mean().alias("cpoe"),
                    pl.col("air_yards").mean().alias("air_yards_per_attempt"),
                    pl.col("yards_after_catch")
                    .filter(pl.col("complete_pass") == 1)
                    .mean()
                    .alias("yac_per_completion"),
                    pl.col("sack").mean().alias("sack_rate"),
                    pl.col("interception").mean().alias("interception_rate"),
                ]
            )
            .sort("game_id")
        )

        if len(game_stats) == 0:
            return self._get_default_passing_features()

        # Compute rolling features
        features = {}
        metrics = [
            "cpoe",
            "air_yards_per_attempt",
            "yac_per_completion",
            "sack_rate",
            "interception_rate",
        ]

        for metric in metrics:
            if metric in game_stats.columns:
                values = game_stats[metric].to_list()
                for window in self.windows:
                    label = self._window_label(window)
                    key = f"{metric}_{label}"
                    features[key] = self._handle_missing(
                        self._calculate_rolling_mean(values, window)
                    )

        return features

    def _get_default_passing_features(self) -> dict[str, float]:
        """Get default values for passing features."""
        defaults = {}
        metrics = [
            "cpoe",
            "air_yards_per_attempt",
            "yac_per_completion",
            "sack_rate",
            "interception_rate",
        ]

        for metric in metrics:
            for window in self.windows:
                label = self._window_label(window)
                defaults[f"{metric}_{label}"] = 0.0

        return defaults

    def _build_rushing_features(
        self,
        pbp_df: pl.DataFrame,
        team: str,
    ) -> dict[str, float]:
        """Build rushing efficiency features."""
        rush_plays = pbp_df.filter(pl.col("play_type") == "run")

        if len(rush_plays) == 0:
            return self._get_default_rushing_features()

        # Compute game-level rushing stats
        game_stats = (
            rush_plays.group_by("game_id")
            .agg(
                [
                    pl.col("yards_gained").mean().alias("yards_per_carry"),
                    (pl.col("epa") > 0).mean().alias("rush_success_rate"),
                    (pl.col("yards_gained") <= 0).mean().alias("stuffed_rate"),
                ]
            )
            .sort("game_id")
        )

        if len(game_stats) == 0:
            return self._get_default_rushing_features()

        # Compute rolling features
        features = {}
        metrics = ["yards_per_carry", "rush_success_rate", "stuffed_rate"]

        for metric in metrics:
            if metric in game_stats.columns:
                values = game_stats[metric].to_list()
                for window in self.windows:
                    label = self._window_label(window)
                    key = f"{metric}_{label}"
                    features[key] = self._handle_missing(
                        self._calculate_rolling_mean(values, window)
                    )

        return features

    def _get_default_rushing_features(self) -> dict[str, float]:
        """Get default values for rushing features."""
        defaults = {}
        metrics = ["yards_per_carry", "rush_success_rate", "stuffed_rate"]

        for metric in metrics:
            for window in self.windows:
                label = self._window_label(window)
                defaults[f"{metric}_{label}"] = 0.0

        return defaults

    def _build_situational_features(
        self,
        pbp_df: pl.DataFrame,
        team: str,
    ) -> dict[str, float]:
        """Build situational efficiency features."""
        if len(pbp_df) == 0:
            return self._get_default_situational_features()

        # Compute game-level situational stats
        game_stats = (
            pbp_df.group_by("game_id")
            .agg(
                [
                    # Red zone TD rate (inside 20)
                    pl.col("touchdown")
                    .filter(pl.col("yardline_100") <= 20)
                    .mean()
                    .alias("red_zone_td_rate"),
                    # Third down conversion rate
                    (
                        (pl.col("yards_gained") >= pl.col("ydstogo"))
                        & (pl.col("down") == 3)
                    )
                    .filter(pl.col("down") == 3)
                    .mean()
                    .alias("third_down_conv_rate"),
                    # First down rate
                    (pl.col("yards_gained") >= pl.col("ydstogo")).mean().alias(
                        "first_down_rate"
                    ),
                ]
            )
            .sort("game_id")
        )

        if len(game_stats) == 0:
            return self._get_default_situational_features()

        # Compute rolling features
        features = {}
        metrics = ["red_zone_td_rate", "third_down_conv_rate", "first_down_rate"]

        for metric in metrics:
            if metric in game_stats.columns:
                values = game_stats[metric].to_list()
                # Handle None values
                values = [v if v is not None else 0.0 for v in values]
                for window in self.windows:
                    label = self._window_label(window)
                    key = f"{metric}_{label}"
                    features[key] = self._handle_missing(
                        self._calculate_rolling_mean(values, window)
                    )

        return features

    def _get_default_situational_features(self) -> dict[str, float]:
        """Get default values for situational features."""
        defaults = {}
        metrics = ["red_zone_td_rate", "third_down_conv_rate", "first_down_rate"]

        for metric in metrics:
            for window in self.windows:
                label = self._window_label(window)
                defaults[f"{metric}_{label}"] = 0.0

        return defaults

    def _build_defensive_features(
        self,
        def_pbp: pl.DataFrame,
        team: str,
    ) -> dict[str, float]:
        """Build defensive efficiency features."""
        if len(def_pbp) == 0:
            return self._get_default_defensive_features()

        # Compute game-level defensive stats
        game_stats = (
            def_pbp.group_by("game_id")
            .agg(
                [
                    pl.col("epa").mean().alias("epa_allowed_per_play"),
                    pl.col("epa")
                    .filter(pl.col("play_type") == "pass")
                    .mean()
                    .alias("epa_allowed_per_pass"),
                    pl.col("epa")
                    .filter(pl.col("play_type") == "run")
                    .mean()
                    .alias("epa_allowed_per_rush"),
                    (pl.col("epa") < 0).mean().alias("defensive_success_rate"),
                ]
            )
            .sort("game_id")
        )

        if len(game_stats) == 0:
            return self._get_default_defensive_features()

        # Compute rolling features
        features = {}
        metrics = [
            "epa_allowed_per_play",
            "epa_allowed_per_pass",
            "epa_allowed_per_rush",
            "defensive_success_rate",
        ]

        for metric in metrics:
            if metric in game_stats.columns:
                values = game_stats[metric].to_list()
                for window in self.windows:
                    label = self._window_label(window)
                    key = f"{metric}_{label}"
                    features[key] = self._handle_missing(
                        self._calculate_rolling_mean(values, window)
                    )

        return features

    def _get_default_defensive_features(self) -> dict[str, float]:
        """Get default values for defensive features."""
        defaults = {}
        metrics = [
            "epa_allowed_per_play",
            "epa_allowed_per_pass",
            "epa_allowed_per_rush",
            "defensive_success_rate",
        ]

        for metric in metrics:
            for window in self.windows:
                label = self._window_label(window)
                defaults[f"{metric}_{label}"] = 0.0

        return defaults

    async def build_matchup_features(
        self,
        pbp_df: pl.DataFrame,
        home_team: str,
        away_team: str,
        season: int,
        week: int,
    ) -> FeatureSet:
        """
        Build features for both teams in a matchup.

        Args:
            pbp_df: Play-by-play DataFrame
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            season: NFL season year
            week: Week number

        Returns:
            FeatureSet with prefixed features for both teams
        """
        # Build features for each team
        home_features = await self.build_features(pbp_df, home_team, season, week)
        away_features = await self.build_features(pbp_df, away_team, season, week)

        # Prefix features with home_/away_
        combined_features = {}

        for key, value in home_features.features.items():
            combined_features[f"home_{key}"] = value

        for key, value in away_features.features.items():
            combined_features[f"away_{key}"] = value

        # Add differential features
        for key in home_features.features:
            home_val = home_features.features.get(key, 0)
            away_val = away_features.features.get(key, 0)
            combined_features[f"diff_{key}"] = home_val - away_val

        return FeatureSet(
            features=combined_features,
            game_id=f"{season}_{week}_{home_team}_vs_{away_team}",
        )


async def build_team_features_for_season(
    pbp_df: pl.DataFrame,
    schedules_df: pl.DataFrame,
    season: int,
    teams: list[str] | None = None,
) -> pl.DataFrame:
    """
    Build team features for all games in a season.

    Useful for creating training datasets.

    Args:
        pbp_df: Play-by-play DataFrame
        schedules_df: Schedules DataFrame
        season: NFL season year
        teams: Optional list of teams to include

    Returns:
        DataFrame with features for each game
    """
    builder = TeamFeatureBuilder()

    # Get all games for the season
    season_games = schedules_df.filter(pl.col("season") == season)

    if teams:
        season_games = season_games.filter(
            (pl.col("home_team").is_in(teams)) | (pl.col("away_team").is_in(teams))
        )

    results = []

    for row in season_games.iter_rows(named=True):
        game_id = row.get("game_id")
        week = row.get("week")
        home_team = row.get("home_team")
        away_team = row.get("away_team")

        if week <= 1:
            continue  # Skip week 1 (no prior data)

        try:
            features = await builder.build_matchup_features(
                pbp_df, home_team, away_team, season, week
            )

            result = {"game_id": game_id, "season": season, "week": week}
            result.update(features.features)
            results.append(result)

        except Exception as e:
            logger.warning(f"Failed to build features for {game_id}: {e}")

    return pl.DataFrame(results) if results else pl.DataFrame()
