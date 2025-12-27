"""
Feature orchestration pipeline.

Unified interface for building all features for spread and prop predictions.
Combines team, player, game, and injury features into ML-ready vectors.
"""
import asyncio
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import polars as pl
from loguru import logger

from .base import FeatureSet, FeatureScaler, fill_missing_features
from .team_features import TeamFeatureBuilder
from .player_features import PlayerFeatureBuilder
from .game_features import GameFeatureBuilder
from .injury_features import (
    InjuryFeatureBuilder,
    calculate_uncertainty_multiplier,
    apply_usage_boosts,
)
from .depth_chart_analyzer import DepthChartAnalyzer


@dataclass
class SpreadPredictionFeatures:
    """Features ready for spread prediction model."""

    game_id: str
    home_team: str
    away_team: str
    features: dict[str, float]
    feature_names: list[str]
    computed_at: datetime

    def to_array(self) -> list[float]:
        """Convert to ordered array for model input."""
        return [self.features.get(name, 0.0) for name in self.feature_names]

    def to_polars(self) -> pl.DataFrame:
        """Convert to single-row Polars DataFrame."""
        return pl.DataFrame([self.features])


@dataclass
class PropPredictionFeatures:
    """Features ready for player prop prediction model."""

    game_id: str
    player_id: str
    player_name: str
    prop_type: str
    features: dict[str, float]
    feature_names: list[str]
    computed_at: datetime

    def to_array(self) -> list[float]:
        """Convert to ordered array for model input."""
        return [self.features.get(name, 0.0) for name in self.feature_names]

    def to_polars(self) -> pl.DataFrame:
        """Convert to single-row Polars DataFrame."""
        return pl.DataFrame([self.features])


@dataclass
class InjuryAdjustedPropFeatures:
    """Features with injury adjustments for prop prediction."""

    game_id: str
    player_id: str
    player_name: str
    prop_type: str
    features: dict[str, float]
    feature_names: list[str]
    computed_at: datetime
    # Injury context
    injury_status: str
    uncertainty_multiplier: float
    is_backup_starter: bool = False
    replacing_player: Optional[str] = None
    usage_boosts_applied: dict[str, float] = None

    def __post_init__(self):
        if self.usage_boosts_applied is None:
            self.usage_boosts_applied = {}

    def to_array(self) -> list[float]:
        """Convert to ordered array for model input."""
        return [self.features.get(name, 0.0) for name in self.feature_names]

    def to_polars(self) -> pl.DataFrame:
        """Convert to single-row Polars DataFrame."""
        return pl.DataFrame([self.features])


class FeaturePipeline:
    """
    Unified feature pipeline for NFL betting predictions.

    Orchestrates all feature builders to produce complete
    feature sets for spread and player prop predictions.

    Example:
        >>> pipeline = FeaturePipeline(data_pipeline)
        >>> await pipeline.initialize()
        >>>
        >>> # Get features for a spread prediction
        >>> features = await pipeline.build_spread_features(
        ...     game_id="2023_05_KC_BUF",
        ...     home_team="BUF",
        ...     away_team="KC",
        ...     season=2023,
        ...     week=5
        ... )
        >>>
        >>> # Get features for a player prop
        >>> prop_features = await pipeline.build_prop_features(
        ...     game_id="2023_05_KC_BUF",
        ...     player_id="00-0036971",
        ...     prop_type="passing_yards",
        ...     season=2023,
        ...     week=5
        ... )
    """

    # Spread model feature names (ordered)
    # Total: ~50 features for comprehensive game prediction
    SPREAD_FEATURE_NAMES = [
        # === CORE OFFENSIVE EPA (14) ===
        # Home team offensive features
        "home_epa_per_play_5g",
        "home_epa_per_pass_5g",
        "home_epa_per_rush_5g",
        "home_success_rate_5g",
        "home_cpoe_5g",
        "home_red_zone_td_rate_5g",
        "home_third_down_conv_rate_5g",
        # Away team offensive features
        "away_epa_per_play_5g",
        "away_epa_per_pass_5g",
        "away_epa_per_rush_5g",
        "away_success_rate_5g",
        "away_cpoe_5g",
        "away_red_zone_td_rate_5g",
        "away_third_down_conv_rate_5g",

        # === CORE DEFENSIVE EPA (4) ===
        # Home team defensive features
        "home_epa_allowed_per_play_5g",
        "home_defensive_success_rate_5g",
        # Away team defensive features
        "away_epa_allowed_per_play_5g",
        "away_defensive_success_rate_5g",

        # === GAME CONTEXT (4) ===
        "rest_advantage",
        "travel_advantage",
        "is_divisional",
        "is_outdoor",

        # === INJURY FEATURES (5) ===
        "health_advantage",
        "home_offense_health",
        "away_offense_health",
        "home_defense_health",
        "away_defense_health",

        # === VEGAS FEATURES (2) ===
        "implied_total",
        "opening_spread",

        # === NEW: OPPONENT-ADJUSTED EPA (3) ===
        # SOS-adjusted efficiency metrics
        "home_adj_epa_per_play_5g",
        "away_adj_epa_per_play_5g",
        "adj_epa_advantage",

        # === NEW: PACE FEATURES (3) ===
        # Tempo and play volume
        "home_plays_per_game_5g",
        "away_plays_per_game_5g",
        "pace_differential",

        # === NEW: CLEAN EPA (garbage-time filtered) (4) ===
        # More predictive for close games
        "home_epa_per_play_clean_5g",
        "away_epa_per_play_clean_5g",
        "home_epa_per_pass_clean_5g",
        "away_epa_per_pass_clean_5g",

        # === NEW: VOLATILITY/CONSISTENCY (4) ===
        # High volatility = less predictable
        "home_epa_volatility_5g",
        "away_epa_volatility_5g",
        "home_yards_volatility_5g",
        "away_yards_volatility_5g",

        # === NEW: ADVANCED PASSING (4) ===
        # Air yards, pressure, and mistakes
        "home_air_yards_per_attempt_5g",
        "away_air_yards_per_attempt_5g",
        "home_sack_rate_5g",
        "away_sack_rate_5g",
    ]

    # Player prop feature names (ordered)
    PROP_FEATURE_NAMES = [
        # Player volume features
        "targets_5g",
        "target_share_5g",
        "receptions_5g",
        "carries_5g",
        "pass_attempts_5g",
        # Player efficiency features
        "yards_per_target_5g",
        "yards_per_carry_5g",
        "yards_per_attempt_5g",
        "receiving_epa_5g",
        "rushing_epa_5g",
        "passing_epa_5g",
        # Usage trends
        "target_trend",
        "usage_trend",
        # Matchup features
        "opp_pass_epa_allowed",
        "opp_rush_epa_allowed",
        # Team context
        "team_implied_score",
        "game_total",
        # Player health
        "player_injury_status",
        "position_group_health",
    ]

    def __init__(
        self,
        data_pipeline=None,
        espn_client=None,
        cache=None,
        cache_ttl_seconds: int = 21600,
    ):
        self.data_pipeline = data_pipeline
        self.espn_client = espn_client
        self.cache = cache
        self.cache_ttl_seconds = cache_ttl_seconds

        # Initialize feature builders
        self.team_builder = TeamFeatureBuilder(cache=cache)
        self.player_builder = PlayerFeatureBuilder(cache=cache)
        self.game_builder = GameFeatureBuilder(cache=cache)
        self.injury_builder = InjuryFeatureBuilder(cache=cache)

        # Initialize depth chart analyzer if ESPN client provided
        self.depth_analyzer: Optional[DepthChartAnalyzer] = None
        if espn_client:
            self.depth_analyzer = DepthChartAnalyzer(espn_client)

        # Feature scaler (fitted during training)
        self.scaler: Optional[FeatureScaler] = None

        self.logger = logger.bind(component="feature_pipeline")
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the pipeline and verify data sources."""
        if self._initialized:
            return

        self.logger.info("Initializing feature pipeline...")

        # Verify data pipeline is available
        if self.data_pipeline:
            health = await self.data_pipeline.health_check()
            self.logger.info(f"Data pipeline status: {health.status}")

        self._initialized = True
        self.logger.info("Feature pipeline initialized")

    async def lookup_player_id(
        self,
        player_name: str,
        season: int,
        position: Optional[str] = None,
    ) -> Optional[str]:
        """
        Look up nflverse player_id from player name.

        Uses roster data as primary source (fast, reliable), falls back to PBP.

        Args:
            player_name: Player's display name (e.g., "Patrick Mahomes")
            season: Season to search in
            position: Optional position hint (QB, RB, WR, TE)

        Returns:
            Player ID (gsis_id) if found, None otherwise
        """
        if not self.data_pipeline:
            return None

        try:
            # Try roster data first - fast and reliable
            seasons_to_try = [season, season - 1, season - 2]
            name_lower = player_name.lower().strip()
            parts = player_name.strip().split()

            for s in seasons_to_try:
                try:
                    roster_df = await self.data_pipeline.get_rosters([s])
                    if roster_df is None or len(roster_df) == 0:
                        continue

                    # Check required columns exist
                    if "full_name" not in roster_df.columns or "gsis_id" not in roster_df.columns:
                        self.logger.debug(f"Roster missing required columns for season {s}")
                        continue

                    # Try exact match on full_name
                    matches = roster_df.filter(
                        pl.col("full_name").str.to_lowercase() == name_lower
                    )
                    if len(matches) > 0:
                        gsis_id = matches["gsis_id"][0]
                        if gsis_id:
                            self.logger.debug(f"Found {player_name} via roster exact match: {gsis_id}")
                            return gsis_id

                    # Try partial match (last name + position)
                    if len(parts) >= 2:
                        last_name = parts[-1].lower()
                        matches = roster_df.filter(
                            pl.col("full_name").str.to_lowercase().str.ends_with(last_name)
                        )
                        # Filter by position if provided
                        if position and "position" in roster_df.columns and len(matches) > 1:
                            pos_matches = matches.filter(pl.col("position") == position.upper())
                            if len(pos_matches) > 0:
                                matches = pos_matches

                        if len(matches) == 1:
                            gsis_id = matches["gsis_id"][0]
                            if gsis_id:
                                self.logger.debug(f"Found {player_name} via roster partial match: {gsis_id}")
                                return gsis_id

                        # If multiple matches and first name initial matches, use that
                        if len(matches) > 1 and len(parts) >= 2:
                            first_initial = parts[0][0].lower()
                            for row in matches.iter_rows(named=True):
                                full = row.get("full_name", "").lower()
                                if full.startswith(first_initial):
                                    gsis_id = row.get("gsis_id")
                                    if gsis_id:
                                        self.logger.debug(f"Found {player_name} via initial match: {gsis_id}")
                                        return gsis_id

                except Exception as e:
                    self.logger.debug(f"Roster lookup failed for season {s}: {e}")
                    continue

            # Fall back to PBP if roster fails
            self.logger.debug(f"Roster lookup failed for {player_name}, trying PBP fallback")
            return await self._lookup_player_id_from_pbp(player_name, season, position)

        except Exception as e:
            self.logger.warning(f"Player lookup failed for {player_name}: {e}")
            return None

    async def _lookup_player_id_from_pbp(
        self,
        player_name: str,
        season: int,
        position: Optional[str] = None,
    ) -> Optional[str]:
        """
        Fallback: Look up player_id from PBP data.

        This is slower but works for players who might not be in roster data.
        """
        if not self.data_pipeline:
            return None

        try:
            # Try multiple seasons for PBP data
            seasons_to_try = [season, season - 1, season - 2]
            pbp_df = None

            for s in seasons_to_try:
                try:
                    pbp_df = await self.data_pipeline.get_historical_pbp([s])
                    if pbp_df is not None and len(pbp_df) > 0:
                        break
                except Exception:
                    continue

            if pbp_df is None or len(pbp_df) == 0:
                return None

            # Convert full name to abbreviated format (e.g., "Patrick Mahomes" -> "P.Mahomes")
            parts = player_name.strip().split()
            if len(parts) >= 2:
                abbrev_name = f"{parts[0][0]}.{parts[-1]}"
            else:
                abbrev_name = player_name

            name_lower = abbrev_name.lower()
            full_name_lower = player_name.lower().strip()

            def try_match(df: pl.DataFrame, name_col: str, id_col: str) -> Optional[str]:
                if name_col not in df.columns or id_col not in df.columns:
                    return None
                # Try abbreviated name first (P.Mahomes)
                matches = df.filter(
                    pl.col(name_col).str.to_lowercase() == name_lower
                ).select(id_col).unique()
                if len(matches) > 0 and matches[id_col][0] is not None:
                    return matches[id_col][0]
                # Try full name
                matches = df.filter(
                    pl.col(name_col).str.to_lowercase() == full_name_lower
                ).select(id_col).unique()
                if len(matches) > 0 and matches[id_col][0] is not None:
                    return matches[id_col][0]
                # Try partial match on last name
                last_name = parts[-1].lower() if len(parts) >= 2 else player_name.lower()
                matches = df.filter(
                    pl.col(name_col).str.to_lowercase().str.ends_with(last_name)
                ).select(id_col).unique()
                if len(matches) == 1 and matches[id_col][0] is not None:
                    return matches[id_col][0]
                return None

            # Search based on position
            if position == "QB":
                result = try_match(pbp_df, "passer_player_name", "passer_id")
                if result:
                    return result
            elif position == "RB":
                result = try_match(pbp_df, "rusher_player_name", "rusher_id")
                if result:
                    return result
            elif position in ("WR", "TE"):
                result = try_match(pbp_df, "receiver_player_name", "receiver_id")
                if result:
                    return result

            # Try all columns
            for id_col, name_col in [
                ("passer_id", "passer_player_name"),
                ("receiver_id", "receiver_player_name"),
                ("rusher_id", "rusher_player_name"),
            ]:
                result = try_match(pbp_df, name_col, id_col)
                if result:
                    return result

            self.logger.debug(f"Player ID not found in PBP for: {player_name}")
            return None

        except Exception as e:
            self.logger.warning(f"PBP lookup failed for {player_name}: {e}")
            return None

    async def build_spread_features(
        self,
        game_id: str,
        home_team: str,
        away_team: str,
        season: int,
        week: int,
        pbp_df: Optional[pl.DataFrame] = None,
        schedules_df: Optional[pl.DataFrame] = None,
        odds_data: Optional[dict] = None,
        weather_data: Optional[dict] = None,
    ) -> SpreadPredictionFeatures:
        """
        Build complete feature set for spread prediction.

        Args:
            game_id: Unique game identifier
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            season: NFL season year
            week: Week number
            pbp_df: Optional play-by-play data (fetched if not provided)
            schedules_df: Optional schedules data
            odds_data: Optional odds data from Odds API
            weather_data: Optional weather data (temp, wind)

        Returns:
            SpreadPredictionFeatures ready for model
        """
        self.logger.debug(f"Building spread features: {home_team} vs {away_team}")

        # Fetch data if not provided - try multiple seasons for reliability
        # This handles the case where season=2025 but nflverse labels data as 2024
        if pbp_df is None and self.data_pipeline:
            seasons_to_try = [season, season - 1, season - 2]
            for s in seasons_to_try:
                try:
                    pbp_df = await self.data_pipeline.get_historical_pbp([s])
                    if pbp_df is not None and len(pbp_df) > 0:
                        self.logger.debug(f"Loaded PBP data from season {s} for spread features")
                        break
                except Exception as e:
                    self.logger.debug(f"Failed to load PBP for season {s}: {e}")
                    continue

        if schedules_df is None and self.data_pipeline:
            seasons_to_try = [season, season - 1, season - 2]
            for s in seasons_to_try:
                try:
                    schedules_df = await self.data_pipeline.get_schedules([s])
                    if schedules_df is not None and len(schedules_df) > 0:
                        self.logger.debug(f"Loaded schedules from season {s} for spread features")
                        break
                except Exception as e:
                    self.logger.debug(f"Failed to load schedules for season {s}: {e}")
                    continue

        features = {}

        # Build team features
        if pbp_df is not None:
            team_features = await self.team_builder.build_matchup_features(
                pbp_df, home_team, away_team, season, week
            )
            features.update(team_features.features)

        # Build game context features
        if schedules_df is not None:
            game_features = await self.game_builder.build_features(
                schedules_df, game_id, home_team, away_team, odds_data, weather_data
            )
            features.update(game_features.features)

        # Build injury features
        injury_features = await self.injury_builder.build_matchup_features(
            home_team, away_team
        )
        features.update(injury_features.features)

        # Fill missing features with defaults
        features = fill_missing_features(features, self.SPREAD_FEATURE_NAMES)

        return SpreadPredictionFeatures(
            game_id=game_id,
            home_team=home_team,
            away_team=away_team,
            features=features,
            feature_names=self.SPREAD_FEATURE_NAMES,
            computed_at=datetime.now(),
        )

    async def build_prop_features(
        self,
        game_id: str,
        player_id: str,
        player_name: str,
        prop_type: str,
        season: int,
        week: int,
        opponent_team: str,
        position: Optional[str] = None,
        pbp_df: Optional[pl.DataFrame] = None,
        odds_data: Optional[dict] = None,
    ) -> PropPredictionFeatures:
        """
        Build complete feature set for player prop prediction.

        Args:
            game_id: Unique game identifier
            player_id: nflverse player ID
            player_name: Player's name
            prop_type: Type of prop (passing_yards, rushing_yards, etc.)
            season: NFL season year
            week: Week number
            opponent_team: Opposing team abbreviation
            position: Player position (auto-detected if not provided)
            pbp_df: Optional play-by-play data
            odds_data: Optional odds data

        Returns:
            PropPredictionFeatures ready for model
        """
        self.logger.debug(f"Building prop features: {player_name} - {prop_type}")

        # Fetch data if not provided - try multiple seasons for reliability
        if pbp_df is None and self.data_pipeline:
            seasons_to_try = [season, season - 1, season - 2]
            for s in seasons_to_try:
                try:
                    pbp_df = await self.data_pipeline.get_historical_pbp([s])
                    if pbp_df is not None and len(pbp_df) > 0:
                        self.logger.debug(f"Loaded PBP data from season {s} for prop features")
                        break
                except Exception as e:
                    self.logger.debug(f"Failed to load PBP for season {s}: {e}")
                    continue

        features = {}

        # Build player features
        if pbp_df is not None:
            player_features = await self.player_builder.build_matchup_features(
                pbp_df, player_id, opponent_team, season, week, position
            )
            features.update(player_features.features)

        # Build player injury features
        injury_features = await self.injury_builder.build_player_injury_features(
            player_id, opponent_team, position or "WR"
        )
        features.update(injury_features.features)

        # Add game context from odds
        if odds_data:
            features["team_implied_score"] = odds_data.get("implied_score", 22.5)
            features["game_total"] = odds_data.get("total", 45.0)
        else:
            features["team_implied_score"] = 22.5
            features["game_total"] = 45.0

        # Fill missing features
        features = fill_missing_features(features, self.PROP_FEATURE_NAMES)

        return PropPredictionFeatures(
            game_id=game_id,
            player_id=player_id,
            player_name=player_name,
            prop_type=prop_type,
            features=features,
            feature_names=self.PROP_FEATURE_NAMES,
            computed_at=datetime.now(),
        )

    async def build_training_dataset(
        self,
        seasons: list[int],
        target: str = "spread",
        include_playoffs: bool = False,
    ) -> pl.DataFrame:
        """
        Build complete training dataset for a given target.

        Args:
            seasons: List of seasons to include
            target: Target variable ("spread" or prop type)
            include_playoffs: Include playoff games

        Returns:
            Polars DataFrame with features and target
        """
        self.logger.info(f"Building training dataset for {target}: {seasons}")

        if not self.data_pipeline:
            raise ValueError("Data pipeline required for building training dataset")

        # Load data
        pbp_df = await self.data_pipeline.get_historical_pbp(seasons)
        schedules_df = await self.data_pipeline.get_schedules(seasons)

        if target == "spread":
            return await self._build_spread_training_data(
                pbp_df, schedules_df, seasons, include_playoffs
            )
        elif target in ["total", "totals"]:
            return await self._build_totals_training_data(
                pbp_df, schedules_df, seasons, include_playoffs
            )
        else:
            # Player prop training data
            return await self._build_prop_training_data(
                pbp_df, schedules_df, seasons, target
            )

    def _extract_odds_from_schedule_row(self, row: dict) -> Optional[dict]:
        """
        Extract Vegas odds from nflverse schedule data.

        NFLverse schedule includes: spread_line, total_line, home_moneyline, away_moneyline
        """
        spread_line = row.get("spread_line")
        total_line = row.get("total_line")

        if spread_line is None and total_line is None:
            return None

        return {
            "home_team": row.get("home_team"),
            "away_team": row.get("away_team"),
            "opening_spread": spread_line if spread_line is not None else 0.0,
            "current_spread": spread_line if spread_line is not None else 0.0,
            "spreads": {
                "outcomes": [
                    {"name": row.get("home_team"), "point": spread_line if spread_line is not None else 0.0}
                ]
            },
            "totals": {
                "outcomes": [
                    {"name": "Over", "point": total_line if total_line is not None else 45.0}
                ]
            },
            "home_moneyline": row.get("home_moneyline", -110),
            "away_moneyline": row.get("away_moneyline", -110),
        }

    def _extract_weather_from_schedule_row(self, row: dict) -> Optional[dict]:
        """
        Extract weather data from nflverse schedule data.

        NFLverse schedule includes: temp, wind, roof
        """
        temp = row.get("temp")
        wind = row.get("wind")
        roof = row.get("roof", "")

        # If indoor or no weather data, return None to use defaults
        is_indoor = roof in ("dome", "closed")
        if is_indoor:
            return {"temperature": 72.0, "wind_speed": 0.0, "precipitation_prob": 0.0}

        # If outdoor but no weather data, return None
        if temp is None and wind is None:
            return None

        return {
            "temperature": float(temp) if temp is not None else 55.0,
            "wind_speed": float(wind) if wind is not None else 8.0,
            "precipitation_prob": 0.0,  # NFLverse doesn't have precip data
        }

    async def _build_spread_training_data(
        self,
        pbp_df: pl.DataFrame,
        schedules_df: pl.DataFrame,
        seasons: list[int],
        include_playoffs: bool,
    ) -> pl.DataFrame:
        """Build training data for spread prediction."""
        results = []

        # Filter completed games
        completed_games = schedules_df.filter(
            (pl.col("season").is_in(seasons))
            & (pl.col("home_score").is_not_null())
            & (pl.col("week") >= 2)  # Need prior data
        )

        if not include_playoffs:
            completed_games = completed_games.filter(pl.col("game_type") == "REG")

        # CRITICAL: Sort chronologically to prevent look-ahead bias
        completed_games = completed_games.sort("season", "week", "game_id")

        total_games = len(completed_games)
        self.logger.info(f"Processing {total_games} games for training")

        for i, row in enumerate(completed_games.iter_rows(named=True)):
            if i % 50 == 0:
                self.logger.debug(f"Processing game {i}/{total_games}")

            try:
                # Extract Vegas odds from schedule data (already available in nflverse)
                odds_data = self._extract_odds_from_schedule_row(row)

                # Extract weather from schedule data
                weather_data = self._extract_weather_from_schedule_row(row)

                features = await self.build_spread_features(
                    game_id=row["game_id"],
                    home_team=row["home_team"],
                    away_team=row["away_team"],
                    season=row["season"],
                    week=row["week"],
                    pbp_df=pbp_df,
                    schedules_df=schedules_df,
                    odds_data=odds_data,
                    weather_data=weather_data,
                )

                # Add target (actual spread: home - away)
                actual_spread = row["home_score"] - row["away_score"]

                result = {
                    "game_id": row["game_id"],
                    "season": row["season"],
                    "week": row["week"],
                    "home_team": row["home_team"],
                    "away_team": row["away_team"],
                    "actual_spread": actual_spread,
                }
                result.update(features.features)
                results.append(result)

            except Exception as e:
                self.logger.warning(f"Failed to process {row['game_id']}: {e}")

        df = pl.DataFrame(results) if results else pl.DataFrame()

        # Add season_week_index for consistent ordering in downstream splits
        if len(df) > 0:
            df = df.with_columns([
                (pl.col("season") * 100 + pl.col("week")).alias("season_week_index")
            ]).sort("season_week_index", "game_id")

        self.logger.info(f"Built training dataset: {len(df)} rows")
        return df

    async def _build_totals_training_data(
        self,
        pbp_df: pl.DataFrame,
        schedules_df: pl.DataFrame,
        seasons: list[int],
        include_playoffs: bool,
    ) -> pl.DataFrame:
        """
        Build training data for totals (over/under) prediction.

        Uses same features as spread model but with total points target.
        """
        results = []

        # Filter completed games
        completed_games = schedules_df.filter(
            (pl.col("season").is_in(seasons))
            & (pl.col("home_score").is_not_null())
            & (pl.col("week") >= 2)  # Need prior data
        )

        if not include_playoffs:
            completed_games = completed_games.filter(pl.col("game_type") == "REG")

        # CRITICAL: Sort chronologically to prevent look-ahead bias
        completed_games = completed_games.sort("season", "week", "game_id")

        total_games = len(completed_games)
        self.logger.info(f"Processing {total_games} games for totals training")

        for i, row in enumerate(completed_games.iter_rows(named=True)):
            if i % 50 == 0:
                self.logger.debug(f"Processing game {i}/{total_games}")

            try:
                # Extract Vegas odds and weather from schedule data
                odds_data = self._extract_odds_from_schedule_row(row)
                weather_data = self._extract_weather_from_schedule_row(row)

                features = await self.build_spread_features(
                    game_id=row["game_id"],
                    home_team=row["home_team"],
                    away_team=row["away_team"],
                    season=row["season"],
                    week=row["week"],
                    pbp_df=pbp_df,
                    schedules_df=schedules_df,
                    odds_data=odds_data,
                    weather_data=weather_data,
                )

                # Target is total points (home + away)
                total_points = row["home_score"] + row["away_score"]

                result = {
                    "game_id": row["game_id"],
                    "season": row["season"],
                    "week": row["week"],
                    "home_team": row["home_team"],
                    "away_team": row["away_team"],
                    "total_points": total_points,
                }
                result.update(features.features)
                results.append(result)

            except Exception as e:
                self.logger.warning(f"Failed to process {row['game_id']}: {e}")

        df = pl.DataFrame(results) if results else pl.DataFrame()

        # Add season_week_index for consistent ordering in downstream splits
        if len(df) > 0:
            df = df.with_columns([
                (pl.col("season") * 100 + pl.col("week")).alias("season_week_index")
            ]).sort("season_week_index", "game_id")

        self.logger.info(f"Built totals training dataset: {len(df)} rows")
        return df

    async def build_prop_training_dataset(
        self,
        pbp_df: pl.DataFrame,
        schedules_df: pl.DataFrame,
        prop_type: str,
    ) -> tuple[pl.DataFrame, pl.Series]:
        """
        Build training dataset for player prop model.

        Args:
            pbp_df: Play-by-play DataFrame
            schedules_df: Schedules DataFrame
            prop_type: Type of prop (passing_yards, rushing_yards, receiving_yards, receptions)

        Returns:
            Tuple of (X features DataFrame, y target Series)
        """
        df = await self._build_prop_training_data(pbp_df, schedules_df, prop_type)

        if len(df) == 0:
            return pl.DataFrame(), pl.Series(name=prop_type, values=[])

        # Split into features and target
        target_col = prop_type
        metadata_cols = ["player_id", "game_id", "season", "week", "position"]
        feature_cols = [c for c in df.columns if c != target_col and c not in metadata_cols]

        X = df.select(feature_cols)
        y = df.get_column(target_col)

        return X, y

    async def _build_prop_training_data(
        self,
        pbp_df: pl.DataFrame,
        schedules_df: pl.DataFrame,
        prop_type: str,
    ) -> pl.DataFrame:
        """
        Build training data for player prop prediction.

        Extracts player-game records from PBP, computes target values,
        and builds features for each player-game.

        Args:
            pbp_df: Play-by-play DataFrame
            schedules_df: Schedules DataFrame
            prop_type: Type of prop (passing_yards, rushing_yards, receiving_yards, receptions)

        Returns:
            DataFrame with features and target column
        """
        self.logger.info(f"Building prop training data for: {prop_type}")

        # Extract player-game stats based on prop type
        player_games = self._extract_player_game_stats(pbp_df, schedules_df, prop_type)

        if len(player_games) == 0:
            self.logger.warning(f"No player-game data found for {prop_type}")
            return pl.DataFrame()

        # CRITICAL: Sort chronologically to prevent look-ahead bias
        player_games = player_games.sort("season", "week", "game_id", "player_id")

        self.logger.info(f"Found {len(player_games)} player-games for {prop_type}")

        # Build features for each player-game
        results = []
        total = len(player_games)

        for i, row in enumerate(player_games.iter_rows(named=True)):
            if i % 100 == 0:
                self.logger.debug(f"Processing player-game {i}/{total}")

            try:
                # Get position for this prop type
                position = self._get_position_for_prop(prop_type)

                # Build features using PlayerFeatureBuilder
                features = await self.player_builder.build_features(
                    pbp_df=pbp_df,
                    player_id=row["player_id"],
                    season=row["season"],
                    week=row["week"],
                    position=position,
                )

                # Combine metadata, features, and target
                result = {
                    "player_id": row["player_id"],
                    "game_id": row["game_id"],
                    "season": row["season"],
                    "week": row["week"],
                    "position": position,
                    prop_type: row["target"],  # The actual stat value
                }
                result.update(features.features)
                results.append(result)

            except Exception as e:
                self.logger.warning(f"Failed to process {row['player_id']} {row['game_id']}: {e}")

        df = pl.DataFrame(results) if results else pl.DataFrame()

        # Add season_week_index for consistent ordering in downstream splits
        if len(df) > 0:
            df = df.with_columns([
                (pl.col("season") * 100 + pl.col("week")).alias("season_week_index")
            ]).sort("season_week_index", "game_id", "player_id")

        self.logger.info(f"Built prop training dataset: {len(df)} rows")
        return df

    def _extract_player_game_stats(
        self,
        pbp_df: pl.DataFrame,
        schedules_df: pl.DataFrame,
        prop_type: str,
    ) -> pl.DataFrame:
        """
        Extract player-game stats from play-by-play data.

        Groups plays by player and game to compute the target stat.

        Returns:
            DataFrame with columns: player_id, game_id, season, week, target
        """
        # Filter to regular season completed games in week 2+
        completed_games = schedules_df.filter(
            (pl.col("home_score").is_not_null())
            & (pl.col("week") >= 2)
            & (pl.col("game_type") == "REG")
        ).select(["game_id", "season", "week"])

        if prop_type == "passing_yards":
            # Sum yards for each passer in each game
            player_stats = (
                pbp_df.filter(
                    (pl.col("passer_id").is_not_null())
                    & (pl.col("play_type") == "pass")
                )
                .group_by(["passer_id", "game_id"])
                .agg(pl.col("yards_gained").sum().alias("target"))
                .rename({"passer_id": "player_id"})
            )

        elif prop_type == "rushing_yards":
            # Sum yards for each rusher in each game
            player_stats = (
                pbp_df.filter(
                    (pl.col("rusher_id").is_not_null())
                    & (pl.col("play_type") == "run")
                )
                .group_by(["rusher_id", "game_id"])
                .agg(pl.col("yards_gained").sum().alias("target"))
                .rename({"rusher_id": "player_id"})
            )

        elif prop_type == "receiving_yards":
            # Sum yards for each receiver in each game (completed passes only)
            player_stats = (
                pbp_df.filter(
                    (pl.col("receiver_id").is_not_null())
                    & (pl.col("complete_pass") == 1)
                )
                .group_by(["receiver_id", "game_id"])
                .agg(pl.col("yards_gained").sum().alias("target"))
                .rename({"receiver_id": "player_id"})
            )

        elif prop_type == "receptions":
            # Count receptions for each receiver in each game
            player_stats = (
                pbp_df.filter(
                    (pl.col("receiver_id").is_not_null())
                    & (pl.col("complete_pass") == 1)
                )
                .group_by(["receiver_id", "game_id"])
                .agg(pl.len().alias("target"))
                .rename({"receiver_id": "player_id"})
            )

        else:
            self.logger.error(f"Unknown prop type: {prop_type}")
            return pl.DataFrame()

        # Filter to minimum stat thresholds to reduce noise
        min_thresholds = {
            "passing_yards": 50,
            "rushing_yards": 10,
            "receiving_yards": 10,
            "receptions": 1,
        }
        min_val = min_thresholds.get(prop_type, 1)
        player_stats = player_stats.filter(pl.col("target") >= min_val)

        # Join with schedule to get season/week
        result = player_stats.join(completed_games, on="game_id", how="inner")

        return result

    def _get_position_for_prop(self, prop_type: str) -> str:
        """Get the primary position for a prop type."""
        positions = {
            "passing_yards": "QB",
            "rushing_yards": "RB",
            "receiving_yards": "WR",
            "receptions": "WR",
        }
        return positions.get(prop_type, "WR")

    def fit_scaler(self, df: pl.DataFrame, feature_cols: list[str]) -> None:
        """
        Fit feature scaler on training data.

        Args:
            df: Training DataFrame
            feature_cols: Columns to scale
        """
        self.scaler = FeatureScaler()
        self.scaler.fit(df, feature_cols)
        self.logger.info(f"Fitted scaler on {len(feature_cols)} features")

    def save_scaler(self, path: Path) -> None:
        """Save fitted scaler to file."""
        if self.scaler:
            self.scaler.save(str(path))
            self.logger.info(f"Saved scaler to {path}")

    def load_scaler(self, path: Path) -> None:
        """Load scaler from file."""
        self.scaler = FeatureScaler()
        self.scaler.load(str(path))
        self.logger.info(f"Loaded scaler from {path}")

    def scale_features(self, features: dict[str, float]) -> dict[str, float]:
        """
        Scale features using fitted scaler.

        Args:
            features: Raw feature values

        Returns:
            Scaled feature values
        """
        if self.scaler is None:
            return features

        df = pl.DataFrame([features])
        scaled_df = self.scaler.standardize(df)
        return scaled_df.row(0, named=True)

    def get_feature_importance_names(self, target: str = "spread") -> list[str]:
        """Get ordered feature names for importance analysis."""
        if target == "spread":
            return self.SPREAD_FEATURE_NAMES
        else:
            return self.PROP_FEATURE_NAMES

    async def build_injury_adjusted_prop_features(
        self,
        game_id: str,
        player_id: str,
        player_name: str,
        prop_type: str,
        season: int,
        week: int,
        opponent_team: str,
        player_team: str,
        position: str,
        pbp_df: Optional[pl.DataFrame] = None,
        odds_data: Optional[dict] = None,
    ) -> InjuryAdjustedPropFeatures:
        """
        Build prop features with full injury context adjustments.

        Includes:
        - Usage boosts from injured teammates
        - Uncertainty multiplier for prediction intervals
        - Backup player role detection

        Args:
            game_id: Unique game identifier
            player_id: ESPN player ID
            player_name: Player's name
            prop_type: Type of prop (passing_yards, etc.)
            season: NFL season year
            week: Week number
            opponent_team: Opposing team abbreviation
            player_team: Player's team abbreviation
            position: Player position
            pbp_df: Optional play-by-play data
            odds_data: Optional odds data

        Returns:
            InjuryAdjustedPropFeatures with uncertainty and usage adjustments
        """
        self.logger.debug(f"Building injury-adjusted features: {player_name}")

        # Build base features
        base_features = await self.build_prop_features(
            game_id=game_id,
            player_id=player_id,
            player_name=player_name,
            prop_type=prop_type,
            season=season,
            week=week,
            opponent_team=opponent_team,
            position=position,
            pbp_df=pbp_df,
            odds_data=odds_data,
        )

        # Default values if no depth analyzer
        injury_status = "ACTIVE"
        uncertainty_multiplier = 1.0
        is_backup_starter = False
        replacing_player = None
        usage_boosts = {}

        if self.depth_analyzer:
            try:
                # Get injury context for this player
                context = await self.depth_analyzer.get_player_injury_context(
                    player_id=player_id,
                    team=player_team,
                    position=position,
                    season=season,
                )

                injury_status = context.get("injury_status", "ACTIVE")
                is_backup_starter = context.get("is_backup_starter", False)
                replacing_player = context.get("replacing_player")
                usage_boosts = context.get("teammate_injury_boosts", {})
                uncertainty_multiplier = context.get("uncertainty_multiplier", 1.0)

            except Exception as e:
                self.logger.warning(f"Failed to get injury context: {e}")

        # Apply usage boosts to features
        adjusted_features = base_features.features.copy()
        if usage_boosts:
            adjusted_features = apply_usage_boosts(adjusted_features, usage_boosts)

        return InjuryAdjustedPropFeatures(
            game_id=game_id,
            player_id=player_id,
            player_name=player_name,
            prop_type=prop_type,
            features=adjusted_features,
            feature_names=base_features.feature_names,
            computed_at=datetime.now(),
            injury_status=injury_status,
            uncertainty_multiplier=uncertainty_multiplier,
            is_backup_starter=is_backup_starter,
            replacing_player=replacing_player,
            usage_boosts_applied=usage_boosts,
        )

    async def get_backup_player_features(
        self,
        team: str,
        season: int,
        week: int,
        opponent_team: str,
        prop_types: list[str],
        pbp_df: Optional[pl.DataFrame] = None,
        odds_data: Optional[dict] = None,
    ) -> list[InjuryAdjustedPropFeatures]:
        """
        Get features for backup players stepping into starter roles.

        Identifies players who are backups filling in for injured starters
        and builds predictions for them with appropriate usage boosts.

        Args:
            team: Team abbreviation
            season: NFL season year
            week: Week number
            opponent_team: Opposing team
            prop_types: List of prop types to generate features for
            pbp_df: Optional play-by-play data
            odds_data: Optional odds data

        Returns:
            List of InjuryAdjustedPropFeatures for backup players
        """
        if not self.depth_analyzer:
            self.logger.warning("Depth analyzer not available for backup predictions")
            return []

        results = []

        try:
            # Analyze injury impact for team
            analysis = await self.depth_analyzer.analyze_injury_impact(team, season)

            # Generate features for each backup activation
            for activation in analysis.backup_activations:
                position = activation.position

                # Determine which prop types apply to this position
                applicable_props = self._get_props_for_position(position, prop_types)

                for prop_type in applicable_props:
                    try:
                        # Build features for the backup player
                        features = await self.build_injury_adjusted_prop_features(
                            game_id=f"{season}_{week}_{team}_{opponent_team}",
                            player_id=activation.backup_player_id,
                            player_name=activation.backup_player_name,
                            prop_type=prop_type,
                            season=season,
                            week=week,
                            opponent_team=opponent_team,
                            player_team=team,
                            position=position,
                            pbp_df=pbp_df,
                            odds_data=odds_data,
                        )

                        # Override with backup-specific values
                        features.is_backup_starter = True
                        features.replacing_player = activation.replacing_player_name
                        features.usage_boosts_applied = activation.expected_usage_boost

                        results.append(features)

                    except Exception as e:
                        self.logger.warning(
                            f"Failed to build backup features for {activation.backup_player_name}: {e}"
                        )

        except Exception as e:
            self.logger.error(f"Failed to analyze injury impact for {team}: {e}")

        return results

    def _get_props_for_position(
        self,
        position: str,
        available_props: list[str],
    ) -> list[str]:
        """Get prop types applicable to a position."""
        position_props = {
            "QB": ["passing_yards"],
            "RB": ["rushing_yards", "receiving_yards", "receptions"],
            "WR": ["receiving_yards", "receptions"],
            "TE": ["receiving_yards", "receptions"],
            "FB": ["rushing_yards", "receiving_yards"],
        }

        applicable = position_props.get(position.upper(), [])
        return [p for p in applicable if p in available_props]


async def create_feature_pipeline(
    data_pipeline=None,
    espn_client=None,
    cache=None,
) -> FeaturePipeline:
    """
    Create and initialize a feature pipeline.

    Args:
        data_pipeline: Optional data pipeline for fetching data
        espn_client: Optional ESPN client for injury/depth data
        cache: Optional cache manager

    Returns:
        Initialized FeaturePipeline
    """
    pipeline = FeaturePipeline(
        data_pipeline=data_pipeline,
        espn_client=espn_client,
        cache=cache,
    )
    await pipeline.initialize()
    return pipeline
