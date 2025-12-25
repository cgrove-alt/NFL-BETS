"""
Data pipeline orchestration layer.

Provides unified access to all data sources with:
- Automatic source initialization
- Health monitoring
- Data validation and normalization
- Team name standardization
- Graceful degradation when sources unavailable
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import polars as pl
from loguru import logger

from .sources.base import DataSourceHealth, DataSourceStatus
from .sources.nflverse import NFLVerseClient
from .sources.odds_api import OddsAPIClient
from .sources.pff_client import PFFClient
from .sources.ftn_dvoa import FTNDVOAClient
from .sources.sic_score import SICScoreClient
from .cache.cache_manager import CacheManager


@dataclass
class PipelineHealth:
    """Overall health status of the data pipeline."""

    status: str  # healthy, degraded, unhealthy
    sources: dict[str, DataSourceHealth]
    timestamp: datetime = field(default_factory=datetime.now)
    message: Optional[str] = None


@dataclass
class GameData:
    """Unified game data from all sources."""

    game_id: str
    home_team: str
    away_team: str
    kickoff: datetime
    week: int
    season: int

    # Odds data
    odds: Optional[dict] = None

    # Team metrics
    home_dvoa: Optional[dict] = None
    away_dvoa: Optional[dict] = None

    # Health data
    home_health: Optional[dict] = None
    away_health: Optional[dict] = None

    # Player data
    home_grades: Optional[pl.DataFrame] = None
    away_grades: Optional[pl.DataFrame] = None


class DataPipeline:
    """
    Unified data access layer for NFL betting system.

    Orchestrates all data sources and provides:
    - Single point of access for all data
    - Automatic caching
    - Health monitoring
    - Error handling with graceful degradation
    - Team name standardization

    Example:
        >>> pipeline = DataPipeline.from_settings(settings)
        >>> await pipeline.initialize()
        >>>
        >>> # Get upcoming games with all data
        >>> games = await pipeline.get_upcoming_games_enriched()
        >>>
        >>> # Get team comparison
        >>> matchup = await pipeline.get_matchup_data("KC", "BUF", season=2023)
    """

    # Team name standardization
    TEAM_NAME_MAP = {
        # Standard abbreviations
        "ARI": "ARI",
        "ATL": "ATL",
        "BAL": "BAL",
        "BUF": "BUF",
        "CAR": "CAR",
        "CHI": "CHI",
        "CIN": "CIN",
        "CLE": "CLE",
        "DAL": "DAL",
        "DEN": "DEN",
        "DET": "DET",
        "GB": "GB",
        "HOU": "HOU",
        "IND": "IND",
        "JAX": "JAX",
        "KC": "KC",
        "LA": "LAR",
        "LAC": "LAC",
        "LAR": "LAR",
        "LV": "LV",
        "MIA": "MIA",
        "MIN": "MIN",
        "NE": "NE",
        "NO": "NO",
        "NYG": "NYG",
        "NYJ": "NYJ",
        "PHI": "PHI",
        "PIT": "PIT",
        "SEA": "SEA",
        "SF": "SF",
        "TB": "TB",
        "TEN": "TEN",
        "WAS": "WAS",
        # Alternate names
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
        # Old team names
        "Oakland Raiders": "LV",
        "San Diego Chargers": "LAC",
        "St. Louis Rams": "LAR",
        "Washington Redskins": "WAS",
        "Washington Football Team": "WAS",
    }

    def __init__(
        self,
        nflverse: Optional[NFLVerseClient] = None,
        odds_api: Optional[OddsAPIClient] = None,
        pff: Optional[PFFClient] = None,
        ftn: Optional[FTNDVOAClient] = None,
        sic: Optional[SICScoreClient] = None,
        cache: Optional[CacheManager] = None,
        data_dir: Optional[Path] = None,
    ):
        self.data_dir = data_dir or Path("data")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Initialize data sources
        self.nflverse = nflverse or NFLVerseClient(
            cache_dir=self.data_dir / "cache" / "nflverse"
        )
        self.odds_api = odds_api
        self.pff = pff
        self.ftn = ftn
        self.sic = sic

        # Cache manager
        self.cache = cache or CacheManager.create_sqlite_cache(
            self.data_dir / "cache" / "cache.db"
        )

        # Logger
        self.logger = logger.bind(component="pipeline")

        # Track initialization
        self._initialized = False

    @classmethod
    def from_settings(cls, settings) -> "DataPipeline":
        """
        Create a DataPipeline from application settings.

        Args:
            settings: Application settings object

        Returns:
            Configured DataPipeline
        """
        data_dir = Path(settings.data_dir) if hasattr(settings, "data_dir") else None

        # Initialize sources based on settings
        nflverse = NFLVerseClient(
            cache_dir=data_dir / "cache" / "nflverse" if data_dir else None,
            enabled=True,
        )

        # Odds API (required for value detection)
        odds_api = None
        if hasattr(settings, "odds_api") and settings.odds_api.api_key:
            odds_api = OddsAPIClient(
                api_key=settings.odds_api.api_key,
                cache_ttl_seconds=settings.odds_api.cache_ttl_seconds,
                regions=settings.odds_api.regions,
                bookmakers=settings.odds_api.default_bookmakers,
            )

        # PFF (optional premium source)
        pff = None
        if hasattr(settings, "premium_data") and settings.premium_data.pff_api_key:
            pff = PFFClient(
                api_key=settings.premium_data.pff_api_key,
                data_dir=data_dir / "pff" if data_dir else None,
            )
        elif hasattr(settings, "premium_data") and settings.premium_data.pff_enabled:
            pff = PFFClient(
                data_dir=data_dir / "pff" if data_dir else None,
            )

        # FTN/DVOA (optional premium source)
        ftn = None
        if hasattr(settings, "premium_data") and settings.premium_data.ftn_api_key:
            ftn = FTNDVOAClient(
                api_key=settings.premium_data.ftn_api_key,
                data_dir=data_dir / "ftn" if data_dir else None,
            )
        elif hasattr(settings, "premium_data") and settings.premium_data.ftn_enabled:
            ftn = FTNDVOAClient(
                data_dir=data_dir / "ftn" if data_dir else None,
            )

        # SIC Score (optional premium source)
        sic = None
        if hasattr(settings, "premium_data") and settings.premium_data.sic_api_key:
            sic = SICScoreClient(
                api_key=settings.premium_data.sic_api_key,
                data_dir=data_dir / "sic" if data_dir else None,
            )
        elif hasattr(settings, "premium_data") and settings.premium_data.sic_score_enabled:
            sic = SICScoreClient(
                data_dir=data_dir / "sic" if data_dir else None,
            )

        # Cache
        cache = CacheManager.create_from_settings(settings)

        return cls(
            nflverse=nflverse,
            odds_api=odds_api,
            pff=pff,
            ftn=ftn,
            sic=sic,
            cache=cache,
            data_dir=data_dir,
        )

    async def initialize(self) -> PipelineHealth:
        """
        Initialize all data sources and verify connectivity.

        Returns:
            PipelineHealth with status of all sources
        """
        self.logger.info("Initializing data pipeline...")

        health_checks = {}

        # Check each source
        sources = [
            ("nflverse", self.nflverse),
            ("odds_api", self.odds_api),
            ("pff", self.pff),
            ("ftn", self.ftn),
            ("sic", self.sic),
        ]

        for name, source in sources:
            if source is None:
                health_checks[name] = DataSourceHealth(
                    source_name=name,
                    status=DataSourceStatus.DISABLED,
                    error_message="Not configured",
                )
            else:
                try:
                    health_checks[name] = await source.health_check()
                except Exception as e:
                    health_checks[name] = DataSourceHealth(
                        source_name=name,
                        status=DataSourceStatus.UNHEALTHY,
                        error_message=str(e),
                    )

        # Determine overall status
        statuses = [h.status for h in health_checks.values()]

        if all(s == DataSourceStatus.HEALTHY for s in statuses):
            overall_status = "healthy"
        elif any(s == DataSourceStatus.UNHEALTHY for s in statuses):
            overall_status = "degraded"
        else:
            overall_status = "degraded"

        # Critical check: nflverse must be available
        if health_checks["nflverse"].status == DataSourceStatus.UNHEALTHY:
            overall_status = "unhealthy"

        self._initialized = True

        pipeline_health = PipelineHealth(
            status=overall_status,
            sources=health_checks,
        )

        self.logger.info(
            f"Pipeline initialized: {overall_status}",
            sources={k: v.status.value for k, v in health_checks.items()},
        )

        return pipeline_health

    async def health_check(self) -> PipelineHealth:
        """Get current health status of all sources."""
        return await self.initialize()

    def standardize_team_name(self, team: str) -> str:
        """
        Standardize team name to consistent abbreviation.

        Args:
            team: Team name or abbreviation

        Returns:
            Standardized team abbreviation
        """
        team = team.strip()
        return self.TEAM_NAME_MAP.get(team, team.upper())

    async def get_upcoming_games(self) -> pl.DataFrame:
        """
        Get upcoming NFL games from nflverse.

        Returns:
            DataFrame with upcoming games
        """
        return await self.nflverse.get_upcoming_games()

    async def get_game_odds(
        self,
        game_id: Optional[str] = None,
        markets: list[str] | None = None,
    ) -> list[dict]:
        """
        Get current odds for NFL games.

        Args:
            game_id: Optional specific game
            markets: Markets to fetch (h2h, spreads, totals)

        Returns:
            List of games with odds
        """
        if self.odds_api is None:
            self.logger.warning("Odds API not configured")
            return []

        if game_id:
            return [await self.odds_api.get_event_odds(game_id, markets)]
        else:
            return await self.odds_api.get_nfl_odds(markets)

    async def get_player_props(
        self,
        game_id: str,
        prop_types: list[str] | None = None,
    ) -> dict:
        """
        Get player prop odds for a game.

        Args:
            game_id: Game ID from Odds API
            prop_types: Types of props to fetch

        Returns:
            Dict with player prop odds
        """
        if self.odds_api is None:
            self.logger.warning("Odds API not configured")
            return {}

        return await self.odds_api.get_player_props(game_id, prop_types)

    async def get_team_dvoa(
        self,
        team: str,
        season: int,
        week: Optional[int] = None,
    ) -> Optional[dict]:
        """
        Get DVOA metrics for a team.

        Args:
            team: Team abbreviation
            season: NFL season year
            week: Optional week number

        Returns:
            DVOA data or None
        """
        if self.ftn is None:
            return None

        team = self.standardize_team_name(team)

        try:
            return await self.ftn.get_team_dvoa(team, season, week)
        except Exception as e:
            self.logger.warning(f"Failed to get DVOA for {team}: {e}")
            return None

    async def get_team_health(
        self,
        team: str,
    ) -> Optional[dict]:
        """
        Get health/injury data for a team.

        Args:
            team: Team abbreviation

        Returns:
            Health data or None
        """
        if self.sic is None:
            return None

        team = self.standardize_team_name(team)

        try:
            return await self.sic.get_team_health(team)
        except Exception as e:
            self.logger.warning(f"Failed to get health for {team}: {e}")
            return None

    async def get_player_grades(
        self,
        team: str,
        season: int,
        week: Optional[int] = None,
    ) -> Optional[pl.DataFrame]:
        """
        Get PFF grades for a team's players.

        Args:
            team: Team abbreviation
            season: NFL season year
            week: Optional week number

        Returns:
            DataFrame with player grades or None
        """
        if self.pff is None:
            return None

        team = self.standardize_team_name(team)

        try:
            return await self.pff.get_team_grades(team, season, week)
        except Exception as e:
            self.logger.warning(f"Failed to get grades for {team}: {e}")
            return None

    async def get_matchup_data(
        self,
        home_team: str,
        away_team: str,
        season: int,
        week: Optional[int] = None,
    ) -> dict:
        """
        Get comprehensive matchup data from all sources.

        Args:
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            season: NFL season year
            week: Optional week number

        Returns:
            Dict with all matchup data
        """
        home = self.standardize_team_name(home_team)
        away = self.standardize_team_name(away_team)

        # Fetch data concurrently
        tasks = {
            "home_dvoa": self.get_team_dvoa(home, season, week),
            "away_dvoa": self.get_team_dvoa(away, season, week),
            "home_health": self.get_team_health(home),
            "away_health": self.get_team_health(away),
            "home_grades": self.get_player_grades(home, season, week),
            "away_grades": self.get_player_grades(away, season, week),
        }

        results = await asyncio.gather(
            *[task for task in tasks.values()],
            return_exceptions=True,
        )

        matchup_data = {
            "home_team": home,
            "away_team": away,
            "season": season,
            "week": week,
        }

        for key, result in zip(tasks.keys(), results):
            if isinstance(result, Exception):
                self.logger.warning(f"Error fetching {key}: {result}")
                matchup_data[key] = None
            else:
                matchup_data[key] = result

        # Calculate DVOA differential if available
        if matchup_data["home_dvoa"] and matchup_data["away_dvoa"]:
            try:
                matchup_data["dvoa_differential"] = (
                    await self.ftn.get_matchup_dvoa_differential(
                        home, away, season, week
                    )
                )
            except Exception:
                matchup_data["dvoa_differential"] = None

        # Calculate health advantage if available
        if matchup_data["home_health"] and matchup_data["away_health"]:
            matchup_data["health_advantage"] = (
                matchup_data["home_health"].get("overall_health", 100)
                - matchup_data["away_health"].get("overall_health", 100)
            )

        return matchup_data

    async def get_upcoming_games_enriched(
        self,
        include_odds: bool = True,
        include_dvoa: bool = True,
        include_health: bool = True,
    ) -> list[GameData]:
        """
        Get upcoming games with enriched data from all sources.

        Args:
            include_odds: Fetch live odds
            include_dvoa: Fetch DVOA metrics
            include_health: Fetch injury data

        Returns:
            List of GameData objects
        """
        # Get base game data
        games = await self.get_upcoming_games()

        if len(games) == 0:
            return []

        # Get current season/week
        current_week = await self.nflverse.get_current_week()
        current_year = datetime.now().year
        if datetime.now().month < 3:
            current_year -= 1

        enriched_games = []

        # Fetch odds for all games at once
        odds_data = {}
        if include_odds and self.odds_api:
            try:
                all_odds = await self.get_game_odds()
                for odds in all_odds:
                    key = f"{odds.get('away_team')}@{odds.get('home_team')}"
                    odds_data[key] = odds
            except Exception as e:
                self.logger.warning(f"Failed to fetch odds: {e}")

        for row in games.iter_rows(named=True):
            home = self.standardize_team_name(row.get("home_team", ""))
            away = self.standardize_team_name(row.get("away_team", ""))

            game_data = GameData(
                game_id=row.get("game_id", f"{away}@{home}"),
                home_team=home,
                away_team=away,
                kickoff=row.get("gameday", datetime.now()),
                week=row.get("week", current_week or 1),
                season=row.get("season", current_year),
            )

            # Attach odds
            odds_key = f"{away}@{home}"
            if odds_key in odds_data:
                game_data.odds = odds_data[odds_key]

            # Fetch DVOA
            if include_dvoa and self.ftn:
                try:
                    game_data.home_dvoa = await self.get_team_dvoa(
                        home, game_data.season, game_data.week
                    )
                    game_data.away_dvoa = await self.get_team_dvoa(
                        away, game_data.season, game_data.week
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to fetch DVOA: {e}")

            # Fetch health
            if include_health and self.sic:
                try:
                    game_data.home_health = await self.get_team_health(home)
                    game_data.away_health = await self.get_team_health(away)
                except Exception as e:
                    self.logger.warning(f"Failed to fetch health: {e}")

            enriched_games.append(game_data)

        return enriched_games

    async def get_historical_pbp(
        self,
        seasons: list[int],
        columns: list[str] | None = None,
    ) -> pl.DataFrame:
        """
        Get historical play-by-play data for model training.

        Args:
            seasons: List of seasons to load
            columns: Optional column filter

        Returns:
            DataFrame with play-by-play data
        """
        return await self.nflverse.load_pbp_data(seasons, columns)

    async def get_player_stats(
        self,
        seasons: list[int],
        stat_type: str = "offense",
    ) -> pl.DataFrame:
        """
        Get aggregated player statistics.

        Args:
            seasons: List of seasons
            stat_type: Type of stats (offense, defense, kicking)

        Returns:
            DataFrame with player stats
        """
        return await self.nflverse.load_player_stats(seasons, stat_type)

    async def get_schedules(
        self,
        seasons: list[int],
    ) -> pl.DataFrame:
        """
        Get game schedules and results.

        Args:
            seasons: List of seasons

        Returns:
            DataFrame with schedule data
        """
        return await self.nflverse.load_schedules(seasons)

    async def get_rosters(
        self,
        seasons: list[int],
    ) -> pl.DataFrame:
        """
        Get team rosters.

        Args:
            seasons: List of seasons

        Returns:
            DataFrame with roster data
        """
        return await self.nflverse.load_rosters(seasons)

    async def refresh_all_cache(self) -> dict:
        """
        Refresh cached data from all sources.

        Returns:
            Dict with refresh status for each source
        """
        results = {}

        # Refresh nflverse data
        try:
            current_year = datetime.now().year
            await self.nflverse.load_schedules([current_year], force_refresh=True)
            results["nflverse"] = "refreshed"
        except Exception as e:
            results["nflverse"] = f"error: {e}"

        # Refresh odds
        if self.odds_api:
            try:
                await self.odds_api.get_nfl_odds()
                results["odds_api"] = "refreshed"
            except Exception as e:
                results["odds_api"] = f"error: {e}"

        return results

    async def close(self) -> None:
        """Close all data source connections."""
        if self.odds_api:
            await self.odds_api.close()
        if self.pff:
            await self.pff.close()
        if self.ftn:
            await self.ftn.close()
        if self.sic:
            await self.sic.close()
        if self.cache:
            await self.cache.close()

        self.logger.info("Data pipeline closed")


# Convenience function for quick pipeline creation
async def create_pipeline(
    odds_api_key: Optional[str] = None,
    data_dir: Optional[Path] = None,
) -> DataPipeline:
    """
    Create and initialize a data pipeline.

    Args:
        odds_api_key: Optional Odds API key
        data_dir: Optional data directory

    Returns:
        Initialized DataPipeline
    """
    odds_api = None
    if odds_api_key:
        odds_api = OddsAPIClient(api_key=odds_api_key)

    pipeline = DataPipeline(
        odds_api=odds_api,
        data_dir=data_dir,
    )

    await pipeline.initialize()
    return pipeline
