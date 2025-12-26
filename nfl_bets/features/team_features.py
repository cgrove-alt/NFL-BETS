"""
Team-level feature engineering.

Computes rolling team performance metrics from play-by-play data:
- EPA (Expected Points Added) metrics
- Passing efficiency (CPOE, air yards)
- Rushing efficiency
- Defensive performance
- Situational efficiency (red zone, third down)
"""
from __future__ import annotations

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

    CRITICAL IMPROVEMENTS:
    - Opponent-adjusted EPA: Raw EPA adjusted for opponent defensive strength
    - Bayesian priors for weeks 1-4: Blends current data with prior season ratings
    - Prevents early-season volatility from dominating predictions

    Key feature categories:
    - EPA metrics: Overall, passing, and rushing efficiency (opponent-adjusted)
    - Success rate: % of plays with positive EPA
    - Explosive plays: Big play rate
    - CPOE: Completion percentage over expected
    - Situational: Red zone, third down, late game
    - Defensive: EPA allowed, pressure rate
    """

    # Feature definitions - now includes opponent-adjusted versions
    OFFENSIVE_FEATURES = [
        "epa_per_play",
        "epa_per_pass",
        "epa_per_rush",
        "adj_epa_per_play",      # Opponent-adjusted EPA
        "adj_epa_per_pass",      # Opponent-adjusted passing EPA
        "adj_epa_per_rush",      # Opponent-adjusted rushing EPA
        "success_rate",
        "explosive_play_rate",
        "negative_play_rate",
    ]

    # League average EPA allowed (used as baseline for opponent adjustment)
    LEAGUE_AVG_EPA_ALLOWED = 0.0  # Baseline - average defense

    # Default prior season ratings (used when no prior data available)
    # Values represent league-average performance
    DEFAULT_PRIOR = {
        "epa_per_play": 0.0,
        "epa_per_pass": 0.1,
        "epa_per_rush": -0.05,
        "success_rate": 0.45,
        "explosive_play_rate": 0.08,
        "negative_play_rate": 0.20,
        "epa_allowed_per_play": 0.0,
        "epa_allowed_per_pass": 0.1,
        "epa_allowed_per_rush": -0.05,
    }

    # Weeks where Bayesian prior blending applies (early season)
    BAYESIAN_BLEND_WEEKS = 4

    # Garbage-time filtered EPA features (excluding blowouts)
    CLEAN_EPA_FEATURES = [
        "epa_per_play_clean",  # Filtered EPA (score diff < 21)
        "epa_per_pass_clean",
        "epa_per_rush_clean",
    ]

    # EMA features (exponential moving average - weights recent games more)
    EMA_FEATURES = [
        "epa_per_play_ema",
        "success_rate_ema",
    ]

    # Volatility features (consistency indicators)
    VOLATILITY_FEATURES = [
        "epa_volatility",  # Coefficient of variation
        "points_volatility",
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

    # Garbage time threshold (score differential)
    GARBAGE_TIME_THRESHOLD = 21  # Filter plays when score diff > 21

    def __init__(
        self,
        windows: list[int] | None = None,
        cache=None,
        cache_ttl_seconds: int = 21600,
        season_priors: dict[int, dict[str, dict[str, float]]] | None = None,
    ):
        """
        Initialize TeamFeatureBuilder.

        Args:
            windows: Rolling window sizes (games)
            cache: Optional cache backend
            cache_ttl_seconds: Cache TTL
            season_priors: Dict mapping season -> team -> metric -> prior_value
                          Used for Bayesian blending in weeks 1-4
        """
        super().__init__(cache=cache, cache_ttl_seconds=cache_ttl_seconds)
        self.windows = windows or self.DEFAULT_WINDOWS
        # Prior season ratings for Bayesian blending
        # Structure: {season: {team: {metric: value}}}
        self.season_priors = season_priors or {}
        # Cache for opponent defensive ratings (computed per season)
        self._opponent_defense_cache: dict[tuple[int, str, int], float] = {}

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

        # Build offensive features (with opponent adjustment)
        off_features = self._build_offensive_features(
            team_pbp, team, season, week, pbp_df
        )
        features.update(off_features)

        # Build clean EPA features (garbage time filtered)
        clean_features = self._build_clean_epa_features(team_pbp, team)
        features.update(clean_features)

        # Build EMA features (momentum/recency weighted)
        ema_features = self._build_ema_features(team_pbp, team)
        features.update(ema_features)

        # Build volatility features (consistency indicators)
        vol_features = self._build_volatility_features(team_pbp, team)
        features.update(vol_features)

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
        season: int = 0,
        week: int = 0,
        full_pbp_df: pl.DataFrame | None = None,
    ) -> dict[str, float]:
        """
        Build offensive EPA and efficiency features with opponent adjustment.

        CRITICAL IMPROVEMENTS:
        1. Opponent-adjusted EPA: Raw EPA - opponent's avg EPA allowed
           This accounts for strength of schedule
        2. Bayesian prior blending for weeks 1-4

        Args:
            pbp_df: Team's offensive plays
            team: Team abbreviation
            season: Current season (for priors)
            week: Current week (for Bayesian blending)
            full_pbp_df: Full PBP for computing opponent strength
        """
        if len(pbp_df) == 0:
            return self._get_default_offensive_features()

        # Compute game-level aggregates WITH opponent info
        game_stats = (
            pbp_df.group_by(["game_id", "defteam"])
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
                    pl.col("week").first().alias("game_week"),
                ]
            )
            .sort("game_id")
        )

        if len(game_stats) == 0:
            return self._get_default_offensive_features()

        # OPPONENT-ADJUSTED EPA CALCULATION
        # For each game, adjust raw EPA by opponent's defensive strength
        adj_epa_per_play = []
        adj_epa_per_pass = []
        adj_epa_per_rush = []

        for row in game_stats.iter_rows(named=True):
            opponent = row.get("defteam")
            game_week = row.get("game_week", week)
            raw_epa = row.get("epa_per_play", 0) or 0
            raw_epa_pass = row.get("epa_per_pass", 0) or 0
            raw_epa_rush = row.get("epa_per_rush", 0) or 0

            # Get opponent's average EPA allowed up to that week
            opp_epa_allowed = self._get_opponent_epa_allowed(
                full_pbp_df, opponent, season, game_week
            )

            # Adjusted EPA = Raw EPA - Opponent's EPA Allowed (relative to league avg)
            # If opponent allows more EPA than average, our adjusted EPA goes down
            adj_epa_per_play.append(raw_epa - opp_epa_allowed)
            adj_epa_per_pass.append(raw_epa_pass - opp_epa_allowed)
            adj_epa_per_rush.append(raw_epa_rush - opp_epa_allowed)

        # Compute rolling features
        features = {}
        raw_metrics = [
            "epa_per_play",
            "epa_per_pass",
            "epa_per_rush",
            "success_rate",
            "explosive_play_rate",
            "negative_play_rate",
        ]

        for metric in raw_metrics:
            values = game_stats[metric].to_list()
            # Handle None values
            values = [v if v is not None else 0.0 for v in values]

            for window in self.windows:
                label = self._window_label(window)
                key = f"{metric}_{label}"
                raw_value = self._calculate_rolling_mean(values, window)

                # Apply Bayesian prior blending for early weeks
                blended_value = self._apply_bayesian_prior(
                    raw_value, metric, team, season, week, len(values)
                )
                features[key] = self._handle_missing(blended_value)

        # Add opponent-adjusted EPA features
        adj_metrics = [
            ("adj_epa_per_play", adj_epa_per_play),
            ("adj_epa_per_pass", adj_epa_per_pass),
            ("adj_epa_per_rush", adj_epa_per_rush),
        ]

        for metric_name, values in adj_metrics:
            for window in self.windows:
                label = self._window_label(window)
                key = f"{metric_name}_{label}"
                raw_value = self._calculate_rolling_mean(values, window)

                # Apply Bayesian prior blending
                base_metric = metric_name.replace("adj_", "")
                blended_value = self._apply_bayesian_prior(
                    raw_value, base_metric, team, season, week, len(values)
                )
                features[key] = self._handle_missing(blended_value)

        return features

    def _get_opponent_epa_allowed(
        self,
        pbp_df: pl.DataFrame | None,
        opponent: str,
        season: int,
        week: int,
    ) -> float:
        """
        Get opponent's average EPA allowed up to this week.

        Returns EPA allowed relative to league average (0.0).
        Positive value = weak defense (allows more EPA than average).
        Negative value = strong defense (allows less EPA than average).
        """
        if pbp_df is None or opponent is None:
            return self.LEAGUE_AVG_EPA_ALLOWED

        # Check cache
        cache_key = (season, opponent, week)
        if cache_key in self._opponent_defense_cache:
            return self._opponent_defense_cache[cache_key]

        try:
            # Get opponent's defensive plays before this week
            opp_def_plays = pbp_df.filter(
                (pl.col("defteam") == opponent)
                & (pl.col("season") == season)
                & (pl.col("week") < week)
                & (pl.col("play_type").is_in(["pass", "run"]))
            )

            if len(opp_def_plays) < 50:  # Need minimum sample
                return self.LEAGUE_AVG_EPA_ALLOWED

            # Compute average EPA allowed
            avg_epa_allowed = opp_def_plays["epa"].mean()
            if avg_epa_allowed is None:
                avg_epa_allowed = self.LEAGUE_AVG_EPA_ALLOWED

            # Cache the result
            self._opponent_defense_cache[cache_key] = float(avg_epa_allowed)
            return float(avg_epa_allowed)

        except Exception as e:
            self.logger.warning(f"Failed to compute opponent EPA for {opponent}: {e}")
            return self.LEAGUE_AVG_EPA_ALLOWED

    def _apply_bayesian_prior(
        self,
        current_value: float | None,
        metric: str,
        team: str,
        season: int,
        week: int,
        n_games: int,
    ) -> float:
        """
        Apply Bayesian prior blending for early season weeks.

        Formula: (Current_Data * Week_Num + Prior * (5 - Week_Num)) / 5

        For weeks 1-4, blends current season data with prior season ratings.
        After week 4, uses only current season data.

        Args:
            current_value: Current rolling average
            metric: Metric name
            team: Team abbreviation
            season: Current season
            week: Current week number
            n_games: Number of games in sample
        """
        if current_value is None:
            current_value = 0.0

        # Only apply blending for weeks 1-4
        if week > self.BAYESIAN_BLEND_WEEKS:
            return current_value

        # Get prior value (previous season's final rating for this team)
        prior = self._get_prior_value(metric, team, season)

        # Bayesian blend formula: weighted average based on week number
        # Week 1: 20% current, 80% prior
        # Week 2: 40% current, 60% prior
        # Week 3: 60% current, 40% prior
        # Week 4: 80% current, 20% prior
        # Week 5+: 100% current
        effective_week = min(week, self.BAYESIAN_BLEND_WEEKS)

        # Also factor in actual number of games played (might be less than week)
        effective_games = min(n_games, effective_week)

        if effective_games == 0:
            return prior

        current_weight = effective_games
        prior_weight = self.BAYESIAN_BLEND_WEEKS + 1 - effective_games

        blended = (current_value * current_weight + prior * prior_weight) / (
            current_weight + prior_weight
        )

        return blended

    def _get_prior_value(
        self,
        metric: str,
        team: str,
        season: int,
    ) -> float:
        """
        Get prior season's final rating for a team/metric.

        Falls back to league average if no prior available.
        """
        prior_season = season - 1

        # Check if we have stored priors
        if prior_season in self.season_priors:
            team_priors = self.season_priors[prior_season].get(team, {})
            if metric in team_priors:
                return team_priors[metric]

        # Fall back to default league average priors
        return self.DEFAULT_PRIOR.get(metric, 0.0)

    def _get_default_offensive_features(self) -> dict[str, float]:
        """Get default values for offensive features (including opponent-adjusted)."""
        defaults = {}
        metrics = [
            "epa_per_play",
            "epa_per_pass",
            "epa_per_rush",
            "adj_epa_per_play",      # Opponent-adjusted
            "adj_epa_per_pass",      # Opponent-adjusted
            "adj_epa_per_rush",      # Opponent-adjusted
            "success_rate",
            "explosive_play_rate",
            "negative_play_rate",
        ]

        for metric in metrics:
            for window in self.windows:
                label = self._window_label(window)
                defaults[f"{metric}_{label}"] = 0.0

        return defaults

    def _build_clean_epa_features(
        self,
        pbp_df: pl.DataFrame,
        team: str,
    ) -> dict[str, float]:
        """
        Build garbage-time filtered EPA features.

        Filters out plays when score differential > GARBAGE_TIME_THRESHOLD.
        These "clean" metrics better represent true team ability.
        """
        if len(pbp_df) == 0:
            return self._get_default_clean_epa_features()

        # Filter out garbage time plays
        # score_differential is typically available in PBP data
        clean_pbp = pbp_df
        if "score_differential" in pbp_df.columns:
            clean_pbp = pbp_df.filter(
                pl.col("score_differential").abs() <= self.GARBAGE_TIME_THRESHOLD
            )
        elif "posteam_score" in pbp_df.columns and "defteam_score" in pbp_df.columns:
            # Calculate score diff if not available
            clean_pbp = pbp_df.filter(
                (pl.col("posteam_score") - pl.col("defteam_score")).abs()
                <= self.GARBAGE_TIME_THRESHOLD
            )

        if len(clean_pbp) == 0:
            return self._get_default_clean_epa_features()

        # Compute game-level clean EPA aggregates
        game_stats = (
            clean_pbp.group_by("game_id")
            .agg(
                [
                    pl.col("epa").mean().alias("epa_per_play_clean"),
                    pl.col("epa")
                    .filter(pl.col("play_type") == "pass")
                    .mean()
                    .alias("epa_per_pass_clean"),
                    pl.col("epa")
                    .filter(pl.col("play_type") == "run")
                    .mean()
                    .alias("epa_per_rush_clean"),
                ]
            )
            .sort("game_id")
        )

        if len(game_stats) == 0:
            return self._get_default_clean_epa_features()

        # Compute rolling features
        features = {}
        metrics = ["epa_per_play_clean", "epa_per_pass_clean", "epa_per_rush_clean"]

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

    def _get_default_clean_epa_features(self) -> dict[str, float]:
        """Get default values for clean EPA features."""
        defaults = {}
        metrics = ["epa_per_play_clean", "epa_per_pass_clean", "epa_per_rush_clean"]

        for metric in metrics:
            for window in self.windows:
                label = self._window_label(window)
                defaults[f"{metric}_{label}"] = 0.0

        return defaults

    def _build_ema_features(
        self,
        pbp_df: pl.DataFrame,
        team: str,
    ) -> dict[str, float]:
        """
        Build Exponential Moving Average features.

        EMA weights recent games more heavily, better capturing
        team momentum and form changes.
        """
        if len(pbp_df) == 0:
            return self._get_default_ema_features()

        # Compute game-level aggregates
        game_stats = (
            pbp_df.group_by("game_id")
            .agg(
                [
                    pl.col("epa").mean().alias("epa_per_play"),
                    (pl.col("epa") > 0).mean().alias("success_rate"),
                ]
            )
            .sort("game_id")
        )

        if len(game_stats) == 0:
            return self._get_default_ema_features()

        features = {}

        # Compute EMA for key metrics
        for metric in ["epa_per_play", "success_rate"]:
            if metric in game_stats.columns:
                values = game_stats[metric].to_list()
                for window in self.windows:
                    label = self._window_label(window)
                    key = f"{metric}_ema_{label}"
                    features[key] = self._handle_missing(
                        self._calculate_ema(values, window)
                    )

        return features

    def _get_default_ema_features(self) -> dict[str, float]:
        """Get default values for EMA features."""
        defaults = {}
        for metric in ["epa_per_play", "success_rate"]:
            for window in self.windows:
                label = self._window_label(window)
                defaults[f"{metric}_ema_{label}"] = 0.0

        return defaults

    def _build_volatility_features(
        self,
        pbp_df: pl.DataFrame,
        team: str,
    ) -> dict[str, float]:
        """
        Build volatility/consistency features.

        High volatility indicates unpredictable performance,
        which affects betting confidence and line value.
        """
        if len(pbp_df) == 0:
            return self._get_default_volatility_features()

        # Compute game-level aggregates
        game_stats = (
            pbp_df.group_by("game_id")
            .agg(
                [
                    pl.col("epa").mean().alias("epa_per_play"),
                    pl.col("yards_gained").sum().alias("total_yards"),
                ]
            )
            .sort("game_id")
        )

        if len(game_stats) < 3:  # Need at least 3 games for meaningful volatility
            return self._get_default_volatility_features()

        features = {}

        # EPA volatility (coefficient of variation)
        epa_values = game_stats["epa_per_play"].to_list()
        for window in self.windows:
            label = self._window_label(window)
            vol = self._calculate_volatility(epa_values, window)
            features[f"epa_volatility_{label}"] = self._handle_missing(vol)

        # Yards volatility
        yards_values = game_stats["total_yards"].to_list()
        for window in self.windows:
            label = self._window_label(window)
            vol = self._calculate_volatility(yards_values, window)
            features[f"yards_volatility_{label}"] = self._handle_missing(vol)

        return features

    def _get_default_volatility_features(self) -> dict[str, float]:
        """Get default values for volatility features."""
        defaults = {}
        for metric in ["epa_volatility", "yards_volatility"]:
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
