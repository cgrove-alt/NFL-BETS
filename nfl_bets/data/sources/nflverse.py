"""
nflreadpy integration for NFL play-by-play and statistics data.

Provides access to:
- Play-by-play data with EPA, WPA, CPOE
- Player statistics (passing, rushing, receiving)
- Team schedules and game information
- Rosters and player information
- Upcoming games for predictions

Data is cached locally as Parquet files for efficient access.
"""
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Union

import polars as pl
from loguru import logger

from .base import CachedDataSource, DataSourceError, DataSourceHealth, DataSourceStatus
from ..cache.cache_manager import CacheManager


class NFLVerseClient(CachedDataSource[pl.DataFrame]):
    """
    Client for accessing NFL data via nflreadpy.

    nflreadpy provides access to the nflverse data ecosystem including:
    - Play-by-play data (1999-present)
    - Player stats aggregated by week
    - Schedules and game results
    - Rosters with player information

    Data is loaded lazily and cached as Parquet files.
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        cache_ttl_hours: int = 24,
        enabled: bool = True,
    ):
        super().__init__(
            source_name="nflverse",
            cache_ttl_seconds=cache_ttl_hours * 3600,
            enabled=enabled,
        )

        self.cache_dir = cache_dir or Path("data/cache/nflverse")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Check if nflreadpy is available
        self._nflreadpy = None
        try:
            import nflreadpy as nfl

            self._nflreadpy = nfl
            self.logger.info("nflreadpy loaded successfully")
        except ImportError:
            self.logger.warning(
                "nflreadpy not available. Install with: "
                "pip install nflreadpy@git+https://github.com/nflverse/nflreadpy"
            )
            self.enabled = False

    async def _fetch_impl(self, *args, **kwargs) -> pl.DataFrame:
        """Implementation handled by specific methods."""
        raise NotImplementedError("Use specific methods like load_pbp_data()")

    async def health_check(self) -> DataSourceHealth:
        """Check if nflreadpy is working."""
        if not self.enabled:
            return DataSourceHealth(
                source_name=self.source_name,
                status=DataSourceStatus.DISABLED,
                error_message="nflreadpy not installed",
            )

        try:
            # Try to load a small amount of data
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None, lambda: self._nflreadpy.load_schedules(seasons=[2023])
            )
            return DataSourceHealth(
                source_name=self.source_name,
                status=DataSourceStatus.HEALTHY,
                last_success=datetime.now(),
            )
        except Exception as e:
            return DataSourceHealth(
                source_name=self.source_name,
                status=DataSourceStatus.UNHEALTHY,
                error_message=str(e),
            )

    def _get_cache_path(self, data_type: str, key: str) -> Path:
        """Get the cache file path for a data type."""
        return self.cache_dir / f"{data_type}_{key}.parquet"

    def _is_cache_valid(self, cache_path: Path, max_age_hours: int = 24) -> bool:
        """Check if cached data is still valid."""
        if not cache_path.exists():
            return False

        modified_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
        age = datetime.now() - modified_time
        return age < timedelta(hours=max_age_hours)

    async def load_pbp_data(
        self,
        seasons: list[int],
        columns: Optional[list[str]] = None,
        force_refresh: bool = False,
    ) -> pl.DataFrame:
        """
        Load play-by-play data for specified seasons.

        Args:
            seasons: List of season years (e.g., [2022, 2023])
            columns: Optional list of columns to load (reduces memory)
            force_refresh: Force reload from source, ignoring cache

        Returns:
            Polars DataFrame with play-by-play data

        Example:
            >>> client = NFLVerseClient()
            >>> pbp = await client.load_pbp_data([2023])
            >>> print(pbp.columns)
        """
        if not self.enabled:
            raise DataSourceError(
                "nflreadpy not available", self.source_name, retry_allowed=False
            )

        cache_key = f"pbp_{'_'.join(map(str, sorted(seasons)))}"
        cache_path = self._get_cache_path("pbp", cache_key)

        # Check cache
        if not force_refresh and self._is_cache_valid(cache_path):
            self.logger.debug(f"Loading PBP from cache: {cache_path}")
            df = pl.read_parquet(cache_path)
            if columns:
                df = df.select([c for c in columns if c in df.columns])
            return df

        # Load from nflreadpy
        self.logger.info(f"Loading PBP data for seasons: {seasons}")

        loop = asyncio.get_event_loop()

        def _load():
            df = self._nflreadpy.load_pbp(seasons=seasons)
            return df

        try:
            start_time = datetime.now()
            df = await loop.run_in_executor(None, _load)
            elapsed = (datetime.now() - start_time).total_seconds()

            self.logger.info(
                f"Loaded {len(df)} plays for {len(seasons)} seasons in {elapsed:.1f}s"
            )

            # Cache to parquet
            df.write_parquet(cache_path)
            self.logger.debug(f"Cached PBP to: {cache_path}")

            if columns:
                df = df.select([c for c in columns if c in df.columns])

            self._record_success(elapsed * 1000)
            return df

        except Exception as e:
            self._record_failure(str(e))
            raise DataSourceError(
                f"Failed to load PBP data: {e}",
                self.source_name,
                original_error=e,
            )

    async def load_player_stats(
        self,
        seasons: list[int],
        stat_type: str = "offense",
        force_refresh: bool = False,
    ) -> pl.DataFrame:
        """
        Load aggregated player statistics by week.

        Args:
            seasons: List of season years
            stat_type: Type of stats - "offense", "defense", "kicking"
            force_refresh: Force reload from source

        Returns:
            Polars DataFrame with player stats
        """
        if not self.enabled:
            raise DataSourceError(
                "nflreadpy not available", self.source_name, retry_allowed=False
            )

        cache_key = f"player_stats_{stat_type}_{'_'.join(map(str, sorted(seasons)))}"
        cache_path = self._get_cache_path("player_stats", cache_key)

        if not force_refresh and self._is_cache_valid(cache_path):
            self.logger.debug(f"Loading player stats from cache: {cache_path}")
            return pl.read_parquet(cache_path)

        self.logger.info(f"Loading {stat_type} player stats for seasons: {seasons}")

        loop = asyncio.get_event_loop()

        def _load():
            return self._nflreadpy.load_player_stats(seasons=seasons)

        try:
            start_time = datetime.now()
            df = await loop.run_in_executor(None, _load)
            elapsed = (datetime.now() - start_time).total_seconds()

            self.logger.info(f"Loaded {len(df)} player stat rows in {elapsed:.1f}s")

            df.write_parquet(cache_path)
            self._record_success(elapsed * 1000)
            return df

        except Exception as e:
            self._record_failure(str(e))
            raise DataSourceError(
                f"Failed to load player stats: {e}",
                self.source_name,
                original_error=e,
            )

    async def load_schedules(
        self,
        seasons: list[int],
        force_refresh: bool = False,
    ) -> pl.DataFrame:
        """
        Load game schedules and results.

        Args:
            seasons: List of season years
            force_refresh: Force reload from source

        Returns:
            Polars DataFrame with schedule data
        """
        if not self.enabled:
            raise DataSourceError(
                "nflreadpy not available", self.source_name, retry_allowed=False
            )

        cache_key = f"schedules_{'_'.join(map(str, sorted(seasons)))}"
        cache_path = self._get_cache_path("schedules", cache_key)

        # Schedules update frequently during season, use shorter cache
        if not force_refresh and self._is_cache_valid(cache_path, max_age_hours=6):
            self.logger.debug(f"Loading schedules from cache: {cache_path}")
            return pl.read_parquet(cache_path)

        self.logger.info(f"Loading schedules for seasons: {seasons}")

        loop = asyncio.get_event_loop()

        def _load():
            return self._nflreadpy.load_schedules(seasons=seasons)

        try:
            start_time = datetime.now()
            df = await loop.run_in_executor(None, _load)
            elapsed = (datetime.now() - start_time).total_seconds()

            self.logger.info(f"Loaded {len(df)} games in {elapsed:.1f}s")

            df.write_parquet(cache_path)
            self._record_success(elapsed * 1000)
            return df

        except Exception as e:
            self._record_failure(str(e))
            raise DataSourceError(
                f"Failed to load schedules: {e}",
                self.source_name,
                original_error=e,
            )

    async def load_rosters(
        self,
        seasons: list[int],
        force_refresh: bool = False,
    ) -> pl.DataFrame:
        """
        Load team rosters with player information.

        Args:
            seasons: List of season years
            force_refresh: Force reload from source

        Returns:
            Polars DataFrame with roster data
        """
        if not self.enabled:
            raise DataSourceError(
                "nflreadpy not available", self.source_name, retry_allowed=False
            )

        cache_key = f"rosters_{'_'.join(map(str, sorted(seasons)))}"
        cache_path = self._get_cache_path("rosters", cache_key)

        if not force_refresh and self._is_cache_valid(cache_path):
            self.logger.debug(f"Loading rosters from cache: {cache_path}")
            return pl.read_parquet(cache_path)

        self.logger.info(f"Loading rosters for seasons: {seasons}")

        loop = asyncio.get_event_loop()

        def _load():
            return self._nflreadpy.load_rosters(seasons=seasons)

        try:
            start_time = datetime.now()
            df = await loop.run_in_executor(None, _load)
            elapsed = (datetime.now() - start_time).total_seconds()

            self.logger.info(f"Loaded {len(df)} roster entries in {elapsed:.1f}s")

            df.write_parquet(cache_path)
            self._record_success(elapsed * 1000)
            return df

        except Exception as e:
            self._record_failure(str(e))
            raise DataSourceError(
                f"Failed to load rosters: {e}",
                self.source_name,
                original_error=e,
            )

    async def load_team_stats(
        self,
        seasons: list[int],
        force_refresh: bool = False,
    ) -> pl.DataFrame:
        """
        Load aggregated team statistics.

        Args:
            seasons: List of season years
            force_refresh: Force reload from source

        Returns:
            Polars DataFrame with team stats
        """
        if not self.enabled:
            raise DataSourceError(
                "nflreadpy not available", self.source_name, retry_allowed=False
            )

        cache_key = f"team_stats_{'_'.join(map(str, sorted(seasons)))}"
        cache_path = self._get_cache_path("team_stats", cache_key)

        if not force_refresh and self._is_cache_valid(cache_path, max_age_hours=12):
            self.logger.debug(f"Loading team stats from cache: {cache_path}")
            return pl.read_parquet(cache_path)

        self.logger.info(f"Loading team stats for seasons: {seasons}")

        loop = asyncio.get_event_loop()

        def _load():
            return self._nflreadpy.load_team_stats(seasons=True)

        try:
            start_time = datetime.now()
            df = await loop.run_in_executor(None, _load)
            elapsed = (datetime.now() - start_time).total_seconds()

            # Filter to requested seasons
            if "season" in df.columns:
                df = df.filter(pl.col("season").is_in(seasons))

            self.logger.info(f"Loaded {len(df)} team stat rows in {elapsed:.1f}s")

            df.write_parquet(cache_path)
            self._record_success(elapsed * 1000)
            return df

        except Exception as e:
            self._record_failure(str(e))
            raise DataSourceError(
                f"Failed to load team stats: {e}",
                self.source_name,
                original_error=e,
            )

    async def get_upcoming_games(self) -> pl.DataFrame:
        """
        Get upcoming games for the current week.

        Returns:
            DataFrame with upcoming games
        """
        current_year = datetime.now().year
        current_month = datetime.now().month

        # Determine season year
        if current_month >= 9:
            season = current_year
        else:
            season = current_year - 1 if current_month < 3 else current_year

        schedules = await self.load_schedules([season], force_refresh=True)

        # Filter to upcoming games
        now = datetime.now()
        upcoming = schedules.filter(
            pl.col("gameday").is_not_null()
            & (pl.col("gameday") >= now.strftime("%Y-%m-%d"))
        )

        return upcoming.sort("gameday", "gametime")

    async def get_current_week(self) -> Optional[int]:
        """Get the current NFL week number."""
        upcoming = await self.get_upcoming_games()

        if len(upcoming) > 0 and "week" in upcoming.columns:
            return upcoming["week"][0]

        return None

    def clear_cache(self, data_type: Optional[str] = None) -> int:
        """
        Clear cached data files.

        Args:
            data_type: Optional type to clear (e.g., "pbp", "schedules").
                      If None, clears all cache.

        Returns:
            Number of files deleted
        """
        count = 0
        pattern = f"{data_type}_*.parquet" if data_type else "*.parquet"

        for cache_file in self.cache_dir.glob(pattern):
            cache_file.unlink()
            count += 1

        self.logger.info(f"Cleared {count} cache files")
        return count

    async def get_game_by_id(self, game_id: str, season: int) -> Optional[dict]:
        """
        Get game information by game ID.

        Args:
            game_id: nflverse game ID
            season: Season year

        Returns:
            Game info dict or None if not found
        """
        schedules = await self.load_schedules([season])

        game = schedules.filter(pl.col("game_id") == game_id)

        if len(game) == 0:
            return None

        return game.row(0, named=True)

    async def get_team_schedule(
        self,
        team: str,
        season: int,
    ) -> pl.DataFrame:
        """
        Get schedule for a specific team.

        Args:
            team: Team abbreviation (e.g., "KC", "BUF")
            season: Season year

        Returns:
            DataFrame with team's games
        """
        schedules = await self.load_schedules([season])

        return schedules.filter(
            (pl.col("home_team") == team) | (pl.col("away_team") == team)
        ).sort("week")

    async def get_player_game_log(
        self,
        player_id: str,
        seasons: list[int],
    ) -> pl.DataFrame:
        """
        Get game-by-game stats for a player.

        Args:
            player_id: nflverse player ID
            seasons: List of season years

        Returns:
            DataFrame with player's game log
        """
        stats = await self.load_player_stats(seasons)

        return stats.filter(pl.col("player_id") == player_id).sort(
            ["season", "week"]
        )

    async def get_latest_completed_week(
        self,
        season: Optional[int] = None,
        force_refresh: bool = True,
    ) -> tuple[Optional[int], Optional[datetime]]:
        """
        Get the most recently completed NFL week.

        This is used for staleness checking - to know when
        new game data is available that models haven't been
        trained on yet.

        Args:
            season: Season year (defaults to current season)
            force_refresh: Force refresh schedules to get latest

        Returns:
            Tuple of (week_number, latest_game_date) or (None, None)
        """
        if season is None:
            current_year = datetime.now().year
            current_month = datetime.now().month
            season = current_year if current_month >= 9 else current_year - 1

        schedules = await self.load_schedules([season], force_refresh=force_refresh)

        # Filter to completed games (have a result)
        completed = schedules.filter(
            pl.col("result").is_not_null()
        )

        if len(completed) == 0:
            return None, None

        # Get the max week and max game date
        max_week = completed.select(pl.col("week").max()).item()

        # Get the latest game date
        if "gameday" in completed.columns:
            latest_date_str = completed.select(pl.col("gameday").max()).item()
            if isinstance(latest_date_str, str):
                latest_date = datetime.fromisoformat(latest_date_str)
            else:
                latest_date = latest_date_str
        else:
            latest_date = None

        return max_week, latest_date

    async def get_games_since_date(
        self,
        since_date: datetime,
        season: Optional[int] = None,
    ) -> pl.DataFrame:
        """
        Get all completed games since a given date.

        Useful for checking what new data is available since
        the last model training.

        Args:
            since_date: Get games completed after this date
            season: Season year (defaults to current season)

        Returns:
            DataFrame with games completed since the date
        """
        if season is None:
            current_year = datetime.now().year
            current_month = datetime.now().month
            season = current_year if current_month >= 9 else current_year - 1

        schedules = await self.load_schedules([season], force_refresh=True)

        # Filter to completed games after the date
        new_games = schedules.filter(
            pl.col("result").is_not_null()
            & (pl.col("gameday") > since_date.strftime("%Y-%m-%d"))
        )

        return new_games.sort("gameday", "gametime")
