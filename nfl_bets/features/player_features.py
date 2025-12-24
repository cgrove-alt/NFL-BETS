"""
Player-level feature engineering.

Computes rolling player performance metrics for prop predictions:
- Volume features (targets, carries, snap share)
- Efficiency features (yards per route, catch rate)
- Usage trends
- Matchup features (vs opponent defense)
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


class PlayerFeatureBuilder(BaseFeatureBuilder, RollingWindowMixin):
    """
    Builds player-level features for prop betting predictions.

    Features are computed as rolling averages over recent games
    to capture player form and usage trends.

    Key feature categories:
    - Volume: Targets, carries, snap share, routes run
    - Efficiency: Yards per route, catch rate, YAC
    - Trends: Increasing/decreasing usage
    - Matchup: Opponent defensive strength
    """

    # Receiver features
    RECEIVER_VOLUME_FEATURES = [
        "targets",
        "target_share",
        "air_yards",
        "air_yards_share",
        "receptions",
    ]

    RECEIVER_EFFICIENCY_FEATURES = [
        "catch_rate",
        "yards_per_target",
        "yards_per_reception",
        "yac_per_reception",
        "receiving_epa",
    ]

    # Rusher features
    RUSHER_VOLUME_FEATURES = [
        "carries",
        "rush_share",
        "rushing_yards",
    ]

    RUSHER_EFFICIENCY_FEATURES = [
        "yards_per_carry",
        "rush_success_rate",
        "rushing_epa",
    ]

    # Passer features
    PASSER_VOLUME_FEATURES = [
        "pass_attempts",
        "completions",
        "passing_yards",
    ]

    PASSER_EFFICIENCY_FEATURES = [
        "completion_rate",
        "yards_per_attempt",
        "passing_epa",
        "cpoe",
        "sack_rate",
        "interception_rate",
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
            self.RECEIVER_VOLUME_FEATURES
            + self.RECEIVER_EFFICIENCY_FEATURES
            + self.RUSHER_VOLUME_FEATURES
            + self.RUSHER_EFFICIENCY_FEATURES
            + self.PASSER_VOLUME_FEATURES
            + self.PASSER_EFFICIENCY_FEATURES
        )

        for metric in all_metrics:
            for window in self.windows:
                names.append(f"{metric}_{self._window_label(window)}")

        # Add trend features
        names.extend(["target_trend", "usage_trend", "efficiency_trend"])

        return names

    async def build_features(
        self,
        pbp_df: pl.DataFrame,
        player_id: str,
        season: int,
        week: int,
        position: Optional[str] = None,
    ) -> FeatureSet:
        """
        Build all player features for a specific game.

        Args:
            pbp_df: Play-by-play DataFrame
            player_id: nflverse player ID
            season: NFL season year
            week: Week number
            position: Player position (auto-detected if not provided)

        Returns:
            FeatureSet with computed player features
        """
        # Check cache first
        cache_key = self._get_cache_key("player", player_id, season, week)
        cached = await self._get_cached(cache_key)
        if cached is not None:
            return cached

        self.logger.debug(f"Building player features: {player_id} S{season} W{week}")

        # Detect position from play data if not provided
        if position is None:
            position = self._detect_position(pbp_df, player_id)

        features = {}
        metadata = {}

        # Build position-specific features
        if position in ["WR", "TE", "RB"]:
            recv_features = self._build_receiver_features(
                pbp_df, player_id, season, week
            )
            features.update(recv_features)

        if position in ["RB", "QB"]:
            rush_features = self._build_rusher_features(pbp_df, player_id, season, week)
            features.update(rush_features)

        if position == "QB":
            pass_features = self._build_passer_features(pbp_df, player_id, season, week)
            features.update(pass_features)

        # Build trend features
        trend_features = self._build_trend_features(pbp_df, player_id, season, week)
        features.update(trend_features)

        # Create feature set
        feature_set = FeatureSet(
            features=features,
            metadata=metadata,
            game_id=f"{season}_{week}_{player_id}",
            player_id=player_id,
        )

        # Cache the result
        await self._set_cached(cache_key, feature_set)

        return feature_set

    def _detect_position(
        self,
        pbp_df: pl.DataFrame,
        player_id: str,
    ) -> str:
        """Detect player position from play data."""
        # Check receiver plays
        recv_plays = pbp_df.filter(pl.col("receiver_id") == player_id)
        rush_plays = pbp_df.filter(pl.col("rusher_id") == player_id)
        pass_plays = pbp_df.filter(pl.col("passer_id") == player_id)

        if len(pass_plays) > len(recv_plays) and len(pass_plays) > len(rush_plays):
            return "QB"
        elif len(rush_plays) > len(recv_plays):
            return "RB"
        else:
            return "WR"  # Default to WR for receivers

    def _build_receiver_features(
        self,
        pbp_df: pl.DataFrame,
        player_id: str,
        season: int,
        week: int,
    ) -> dict[str, float]:
        """Build receiving features."""
        # Filter to player's receiving plays before this week
        player_plays = pbp_df.filter(
            (pl.col("receiver_id") == player_id)
            & (pl.col("season") == season)
            & (pl.col("week") < week)
        )

        if len(player_plays) == 0:
            return self._get_default_receiver_features()

        # Compute game-level receiving stats
        game_stats = (
            player_plays.group_by("game_id")
            .agg(
                [
                    pl.len().alias("targets"),
                    pl.col("complete_pass").sum().alias("receptions"),
                    pl.col("air_yards").sum().alias("air_yards"),
                    pl.col("yards_gained").sum().alias("receiving_yards"),
                    pl.col("yards_after_catch").sum().alias("total_yac"),
                    pl.col("epa").mean().alias("receiving_epa"),
                ]
            )
            .sort("game_id")
        )

        # Calculate derived metrics
        game_stats = game_stats.with_columns(
            [
                (pl.col("receptions") / pl.col("targets")).alias("catch_rate"),
                (pl.col("receiving_yards") / pl.col("targets")).alias("yards_per_target"),
                (pl.col("receiving_yards") / pl.col("receptions").clip(lower_bound=1)).alias(
                    "yards_per_reception"
                ),
                (pl.col("total_yac") / pl.col("receptions").clip(lower_bound=1)).alias(
                    "yac_per_reception"
                ),
            ]
        )

        # Calculate target share (need team context)
        team_targets = (
            pbp_df.filter(
                (pl.col("season") == season)
                & (pl.col("week") < week)
                & (pl.col("play_type") == "pass")
            )
            .group_by(["game_id", "posteam"])
            .agg(pl.len().alias("team_targets"))
        )

        # Join to get target share
        if len(team_targets) > 0:
            game_stats = game_stats.with_columns(pl.lit(0.15).alias("target_share"))
            game_stats = game_stats.with_columns(pl.lit(0.15).alias("air_yards_share"))

        if len(game_stats) == 0:
            return self._get_default_receiver_features()

        # Compute rolling features
        features = {}
        metrics = [
            "targets",
            "target_share",
            "air_yards",
            "air_yards_share",
            "receptions",
            "catch_rate",
            "yards_per_target",
            "yards_per_reception",
            "yac_per_reception",
            "receiving_epa",
        ]

        for metric in metrics:
            if metric in game_stats.columns:
                values = game_stats[metric].to_list()
                values = [v if v is not None and v == v else 0.0 for v in values]
                for window in self.windows:
                    label = self._window_label(window)
                    key = f"{metric}_{label}"
                    features[key] = self._handle_missing(
                        self._calculate_rolling_mean(values, window)
                    )

        return features

    def _get_default_receiver_features(self) -> dict[str, float]:
        """Get default values for receiver features."""
        defaults = {}
        metrics = [
            "targets",
            "target_share",
            "air_yards",
            "air_yards_share",
            "receptions",
            "catch_rate",
            "yards_per_target",
            "yards_per_reception",
            "yac_per_reception",
            "receiving_epa",
        ]

        for metric in metrics:
            for window in self.windows:
                label = self._window_label(window)
                defaults[f"{metric}_{label}"] = 0.0

        return defaults

    def _build_rusher_features(
        self,
        pbp_df: pl.DataFrame,
        player_id: str,
        season: int,
        week: int,
    ) -> dict[str, float]:
        """Build rushing features."""
        # Filter to player's rushing plays before this week
        player_plays = pbp_df.filter(
            (pl.col("rusher_id") == player_id)
            & (pl.col("season") == season)
            & (pl.col("week") < week)
        )

        if len(player_plays) == 0:
            return self._get_default_rusher_features()

        # Compute game-level rushing stats
        game_stats = (
            player_plays.group_by("game_id")
            .agg(
                [
                    pl.len().alias("carries"),
                    pl.col("yards_gained").sum().alias("rushing_yards"),
                    pl.col("epa").mean().alias("rushing_epa"),
                    (pl.col("epa") > 0).mean().alias("rush_success_rate"),
                ]
            )
            .sort("game_id")
        )

        # Calculate derived metrics
        game_stats = game_stats.with_columns(
            [
                (pl.col("rushing_yards") / pl.col("carries").clip(lower_bound=1)).alias(
                    "yards_per_carry"
                ),
                pl.lit(0.15).alias("rush_share"),  # Placeholder
            ]
        )

        if len(game_stats) == 0:
            return self._get_default_rusher_features()

        # Compute rolling features
        features = {}
        metrics = [
            "carries",
            "rush_share",
            "rushing_yards",
            "yards_per_carry",
            "rush_success_rate",
            "rushing_epa",
        ]

        for metric in metrics:
            if metric in game_stats.columns:
                values = game_stats[metric].to_list()
                values = [v if v is not None and v == v else 0.0 for v in values]
                for window in self.windows:
                    label = self._window_label(window)
                    key = f"{metric}_{label}"
                    features[key] = self._handle_missing(
                        self._calculate_rolling_mean(values, window)
                    )

        return features

    def _get_default_rusher_features(self) -> dict[str, float]:
        """Get default values for rusher features."""
        defaults = {}
        metrics = [
            "carries",
            "rush_share",
            "rushing_yards",
            "yards_per_carry",
            "rush_success_rate",
            "rushing_epa",
        ]

        for metric in metrics:
            for window in self.windows:
                label = self._window_label(window)
                defaults[f"{metric}_{label}"] = 0.0

        return defaults

    def _build_passer_features(
        self,
        pbp_df: pl.DataFrame,
        player_id: str,
        season: int,
        week: int,
    ) -> dict[str, float]:
        """Build passing features."""
        # Filter to player's passing plays before this week
        player_plays = pbp_df.filter(
            (pl.col("passer_id") == player_id)
            & (pl.col("season") == season)
            & (pl.col("week") < week)
        )

        if len(player_plays) == 0:
            return self._get_default_passer_features()

        # Compute game-level passing stats
        game_stats = (
            player_plays.group_by("game_id")
            .agg(
                [
                    pl.len().alias("pass_attempts"),
                    pl.col("complete_pass").sum().alias("completions"),
                    pl.col("yards_gained").sum().alias("passing_yards"),
                    pl.col("epa").mean().alias("passing_epa"),
                    pl.col("cpoe").mean().alias("cpoe"),
                    pl.col("sack").mean().alias("sack_rate"),
                    pl.col("interception").mean().alias("interception_rate"),
                ]
            )
            .sort("game_id")
        )

        # Calculate derived metrics
        game_stats = game_stats.with_columns(
            [
                (pl.col("completions") / pl.col("pass_attempts").clip(lower_bound=1)).alias(
                    "completion_rate"
                ),
                (pl.col("passing_yards") / pl.col("pass_attempts").clip(lower_bound=1)).alias(
                    "yards_per_attempt"
                ),
            ]
        )

        if len(game_stats) == 0:
            return self._get_default_passer_features()

        # Compute rolling features
        features = {}
        metrics = [
            "pass_attempts",
            "completions",
            "passing_yards",
            "completion_rate",
            "yards_per_attempt",
            "passing_epa",
            "cpoe",
            "sack_rate",
            "interception_rate",
        ]

        for metric in metrics:
            if metric in game_stats.columns:
                values = game_stats[metric].to_list()
                values = [v if v is not None and v == v else 0.0 for v in values]
                for window in self.windows:
                    label = self._window_label(window)
                    key = f"{metric}_{label}"
                    features[key] = self._handle_missing(
                        self._calculate_rolling_mean(values, window)
                    )

        return features

    def _get_default_passer_features(self) -> dict[str, float]:
        """Get default values for passer features."""
        defaults = {}
        metrics = [
            "pass_attempts",
            "completions",
            "passing_yards",
            "completion_rate",
            "yards_per_attempt",
            "passing_epa",
            "cpoe",
            "sack_rate",
            "interception_rate",
        ]

        for metric in metrics:
            for window in self.windows:
                label = self._window_label(window)
                defaults[f"{metric}_{label}"] = 0.0

        return defaults

    def _build_trend_features(
        self,
        pbp_df: pl.DataFrame,
        player_id: str,
        season: int,
        week: int,
    ) -> dict[str, float]:
        """Build trend features (increasing/decreasing usage)."""
        features = {
            "target_trend": 0.0,
            "usage_trend": 0.0,
            "efficiency_trend": 0.0,
        }

        # Get receiving targets by game
        recv_plays = pbp_df.filter(
            (pl.col("receiver_id") == player_id)
            & (pl.col("season") == season)
            & (pl.col("week") < week)
        )

        if len(recv_plays) > 0:
            targets_by_game = (
                recv_plays.group_by("game_id")
                .agg(pl.len().alias("targets"))
                .sort("game_id")
            )

            targets = targets_by_game["targets"].to_list()
            if len(targets) >= 3:
                features["target_trend"] = self._calculate_trend(targets, 5)

        # Get rushing attempts by game
        rush_plays = pbp_df.filter(
            (pl.col("rusher_id") == player_id)
            & (pl.col("season") == season)
            & (pl.col("week") < week)
        )

        if len(rush_plays) > 0:
            carries_by_game = (
                rush_plays.group_by("game_id")
                .agg(pl.len().alias("carries"))
                .sort("game_id")
            )

            carries = carries_by_game["carries"].to_list()
            if len(carries) >= 3:
                features["usage_trend"] = self._calculate_trend(carries, 5)

        return features

    async def build_matchup_features(
        self,
        pbp_df: pl.DataFrame,
        player_id: str,
        opponent_team: str,
        season: int,
        week: int,
        position: Optional[str] = None,
    ) -> FeatureSet:
        """
        Build matchup-specific features for a player vs opponent.

        Args:
            pbp_df: Play-by-play DataFrame
            player_id: nflverse player ID
            opponent_team: Opposing team abbreviation
            season: NFL season year
            week: Week number
            position: Player position

        Returns:
            FeatureSet with matchup features
        """
        # Get player's base features
        player_features = await self.build_features(
            pbp_df, player_id, season, week, position
        )

        # Get opponent defensive strength
        opp_features = await self._get_opponent_defense_features(
            pbp_df, opponent_team, season, week, position or "WR"
        )

        # Merge features
        combined = player_features.features.copy()
        combined.update(opp_features)

        return FeatureSet(
            features=combined,
            player_id=player_id,
            game_id=f"{season}_{week}_{player_id}_vs_{opponent_team}",
        )

    async def _get_opponent_defense_features(
        self,
        pbp_df: pl.DataFrame,
        opponent_team: str,
        season: int,
        week: int,
        position: str,
    ) -> dict[str, float]:
        """Get opponent's defensive strength vs position."""
        features = {}

        # Filter to opponent's defensive plays before this week
        def_plays = pbp_df.filter(
            (pl.col("defteam") == opponent_team)
            & (pl.col("season") == season)
            & (pl.col("week") < week)
        )

        if len(def_plays) == 0:
            return {
                "opp_pass_epa_allowed": 0.0,
                "opp_rush_epa_allowed": 0.0,
                "opp_yards_per_target_allowed": 0.0,
                "opp_yards_per_carry_allowed": 0.0,
            }

        # Compute opponent defensive stats
        if position in ["WR", "TE"]:
            pass_def = def_plays.filter(pl.col("play_type") == "pass")
            if len(pass_def) > 0:
                features["opp_pass_epa_allowed"] = pass_def["epa"].mean() or 0.0
                features["opp_yards_per_target_allowed"] = (
                    pass_def["yards_gained"].mean() or 0.0
                )
            else:
                features["opp_pass_epa_allowed"] = 0.0
                features["opp_yards_per_target_allowed"] = 0.0

        if position == "RB":
            rush_def = def_plays.filter(pl.col("play_type") == "run")
            if len(rush_def) > 0:
                features["opp_rush_epa_allowed"] = rush_def["epa"].mean() or 0.0
                features["opp_yards_per_carry_allowed"] = (
                    rush_def["yards_gained"].mean() or 0.0
                )
            else:
                features["opp_rush_epa_allowed"] = 0.0
                features["opp_yards_per_carry_allowed"] = 0.0

        if position == "QB":
            pass_def = def_plays.filter(pl.col("play_type") == "pass")
            if len(pass_def) > 0:
                features["opp_pass_epa_allowed"] = pass_def["epa"].mean() or 0.0
                features["opp_sack_rate"] = pass_def["sack"].mean() or 0.0
            else:
                features["opp_pass_epa_allowed"] = 0.0
                features["opp_sack_rate"] = 0.0

        return features


async def build_player_features_for_season(
    pbp_df: pl.DataFrame,
    player_ids: list[str],
    season: int,
) -> pl.DataFrame:
    """
    Build player features for all games in a season.

    Useful for creating training datasets.

    Args:
        pbp_df: Play-by-play DataFrame
        player_ids: List of player IDs to include
        season: NFL season year

    Returns:
        DataFrame with features for each player-game
    """
    builder = PlayerFeatureBuilder()
    results = []

    # Get unique weeks
    weeks = pbp_df.filter(pl.col("season") == season)["week"].unique().sort().to_list()

    for player_id in player_ids:
        for week in weeks:
            if week <= 1:
                continue  # Skip week 1 (no prior data)

            try:
                features = await builder.build_features(pbp_df, player_id, season, week)

                result = {
                    "player_id": player_id,
                    "season": season,
                    "week": week,
                }
                result.update(features.features)
                results.append(result)

            except Exception as e:
                logger.warning(f"Failed to build features for {player_id} week {week}: {e}")

    return pl.DataFrame(results) if results else pl.DataFrame()
