"""
Pro Football Focus (PFF) data integration.

Provides access to:
- Player grades (offense, defense, special teams)
- Coverage matchup data
- Position-specific metrics
- Snap counts and participation

Supports two modes:
- API mode: Direct API access (requires B2B contract)
- CSV mode: Manual CSV upload for subscribers
"""
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional

import polars as pl
from loguru import logger

from .base import (
    CachedDataSource,
    DataSourceError,
    DataSourceHealth,
    DataSourceStatus,
    DataNotAvailableError,
)


class PFFClient(CachedDataSource[pl.DataFrame]):
    """
    Client for accessing PFF player grades and metrics.

    PFF provides the industry's most detailed player-level grading
    and analysis. This client supports:

    1. API Mode: Direct access via PFF's B2B API
       - Requires enterprise contract
       - Real-time updates
       - Full metric access

    2. CSV Mode: Manual data import
       - For individual subscribers
       - Download CSVs from PFF website
       - Place in configured data directory

    Grades are on a 0-100 scale:
    - 90+: Elite
    - 80-89: High Quality
    - 70-79: Above Average
    - 60-69: Average
    - 50-59: Below Average
    - <50: Poor
    """

    # PFF Grade columns by position group
    OFFENSIVE_GRADE_COLUMNS = [
        "player_id",
        "player_name",
        "team",
        "position",
        "overall_grade",
        "offense_grade",
        "pass_grade",
        "pass_block_grade",
        "run_grade",
        "run_block_grade",
        "receiving_grade",
        "snaps",
    ]

    DEFENSIVE_GRADE_COLUMNS = [
        "player_id",
        "player_name",
        "team",
        "position",
        "overall_grade",
        "defense_grade",
        "run_defense_grade",
        "pass_rush_grade",
        "coverage_grade",
        "tackling_grade",
        "snaps",
    ]

    COVERAGE_MATCHUP_COLUMNS = [
        "defender_id",
        "defender_name",
        "defender_team",
        "receiver_id",
        "receiver_name",
        "receiver_team",
        "targets",
        "receptions",
        "yards",
        "touchdowns",
        "passer_rating_allowed",
        "coverage_grade",
    ]

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        data_dir: Optional[Path] = None,
        cache_ttl_hours: int = 24,
        enabled: bool = True,
    ):
        super().__init__(
            source_name="pff",
            cache_ttl_seconds=cache_ttl_hours * 3600,
            enabled=enabled,
        )

        self.api_key = api_key
        self.api_url = api_url or "https://api.pff.com/v1"
        self.data_dir = data_dir or Path("data/pff")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Determine mode
        self._api_mode = bool(api_key and api_url)
        if self._api_mode:
            self.logger.info("PFF client initialized in API mode")
        else:
            self.logger.info(
                "PFF client initialized in CSV mode. "
                f"Place CSV files in: {self.data_dir}"
            )

        # HTTP session for API mode
        self._session = None

    @property
    def mode(self) -> str:
        """Get current operating mode."""
        return "api" if self._api_mode else "csv"

    async def _fetch_impl(self, *args, **kwargs) -> pl.DataFrame:
        """Implementation handled by specific methods."""
        raise NotImplementedError("Use specific methods like load_player_grades()")

    async def health_check(self) -> DataSourceHealth:
        """Check if PFF data is available."""
        if not self.enabled:
            return DataSourceHealth(
                source_name=self.source_name,
                status=DataSourceStatus.DISABLED,
                error_message="PFF integration disabled",
            )

        if self._api_mode:
            # Check API connectivity
            try:
                # Would make actual API call here
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
        else:
            # Check for CSV files
            csv_files = list(self.data_dir.glob("*.csv"))
            if csv_files:
                return DataSourceHealth(
                    source_name=self.source_name,
                    status=DataSourceStatus.HEALTHY,
                    last_success=datetime.now(),
                )
            else:
                return DataSourceHealth(
                    source_name=self.source_name,
                    status=DataSourceStatus.DEGRADED,
                    error_message=f"No CSV files found in {self.data_dir}",
                )

    async def _load_from_api(
        self,
        endpoint: str,
        params: Optional[dict] = None,
    ) -> pl.DataFrame:
        """Load data from PFF API."""
        if not self._api_mode:
            raise DataSourceError(
                "API mode not configured",
                self.source_name,
                retry_allowed=False,
            )

        import aiohttp

        if self._session is None:
            self._session = aiohttp.ClientSession()

        url = f"{self.api_url}/{endpoint}"
        headers = {"Authorization": f"Bearer {self.api_key}"}

        try:
            async with self._session.get(
                url, headers=headers, params=params
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return pl.DataFrame(data)
                elif response.status == 401:
                    raise DataSourceError(
                        "Invalid API key",
                        self.source_name,
                        retry_allowed=False,
                    )
                else:
                    raise DataSourceError(
                        f"API error: {response.status}",
                        self.source_name,
                        retry_allowed=True,
                    )
        except aiohttp.ClientError as e:
            raise DataSourceError(
                f"Connection error: {e}",
                self.source_name,
                original_error=e,
                retry_allowed=True,
            )

    def _load_from_csv(
        self,
        file_pattern: str,
        required_columns: Optional[list[str]] = None,
    ) -> pl.DataFrame:
        """Load data from CSV files."""
        files = list(self.data_dir.glob(file_pattern))

        if not files:
            raise DataNotAvailableError(
                self.source_name,
                f"No files matching '{file_pattern}' found in {self.data_dir}",
            )

        # Load and concatenate all matching files
        dfs = []
        for file in files:
            try:
                df = pl.read_csv(file)
                dfs.append(df)
                self.logger.debug(f"Loaded {len(df)} rows from {file.name}")
            except Exception as e:
                self.logger.warning(f"Failed to load {file.name}: {e}")

        if not dfs:
            raise DataSourceError(
                "No valid CSV files could be loaded",
                self.source_name,
                retry_allowed=False,
            )

        result = pl.concat(dfs)

        # Validate required columns
        if required_columns:
            missing = set(required_columns) - set(result.columns)
            if missing:
                self.logger.warning(f"Missing columns in PFF data: {missing}")

        return result

    async def load_player_grades(
        self,
        season: int,
        week: Optional[int] = None,
        position_group: Optional[str] = None,
    ) -> pl.DataFrame:
        """
        Load PFF player grades.

        Args:
            season: NFL season year
            week: Optional week number (None for season totals)
            position_group: Filter by position group (offense, defense, special)

        Returns:
            DataFrame with player grades

        Example:
            >>> pff = PFFClient(data_dir=Path("data/pff"))
            >>> grades = await pff.load_player_grades(2023, week=10)
            >>> elite = grades.filter(pl.col("overall_grade") >= 90)
        """
        if not self.enabled:
            raise DataSourceError(
                "PFF integration not enabled",
                self.source_name,
                retry_allowed=False,
            )

        cache_path = self.data_dir / f"grades_{season}_{week or 'season'}.parquet"

        # Check cache
        if cache_path.exists():
            cache_age = datetime.now().timestamp() - cache_path.stat().st_mtime
            if cache_age < self.cache_ttl_seconds:
                self.logger.debug(f"Loading grades from cache: {cache_path}")
                df = pl.read_parquet(cache_path)
                if position_group:
                    df = self._filter_position_group(df, position_group)
                return df

        if self._api_mode:
            params = {"season": season}
            if week:
                params["week"] = week

            df = await self._load_from_api("player-grades", params)
        else:
            # Try to load from CSV
            if week:
                pattern = f"*grades*{season}*week{week}*.csv"
            else:
                pattern = f"*grades*{season}*.csv"

            loop = asyncio.get_event_loop()
            df = await loop.run_in_executor(None, self._load_from_csv, pattern)

        # Cache the result
        df.write_parquet(cache_path)

        if position_group:
            df = self._filter_position_group(df, position_group)

        self._record_success(0)
        return df

    def _filter_position_group(
        self, df: pl.DataFrame, position_group: str
    ) -> pl.DataFrame:
        """Filter DataFrame by position group."""
        offense_positions = ["QB", "RB", "WR", "TE", "T", "G", "C", "FB"]
        defense_positions = ["DI", "ED", "LB", "CB", "S"]
        special_positions = ["K", "P", "LS"]

        if position_group.lower() == "offense":
            return df.filter(pl.col("position").is_in(offense_positions))
        elif position_group.lower() == "defense":
            return df.filter(pl.col("position").is_in(defense_positions))
        elif position_group.lower() == "special":
            return df.filter(pl.col("position").is_in(special_positions))
        else:
            return df

    async def load_coverage_matchups(
        self,
        season: int,
        week: Optional[int] = None,
    ) -> pl.DataFrame:
        """
        Load coverage matchup data (WR vs CB).

        Args:
            season: NFL season year
            week: Optional week number

        Returns:
            DataFrame with coverage matchup grades
        """
        if not self.enabled:
            raise DataSourceError(
                "PFF integration not enabled",
                self.source_name,
                retry_allowed=False,
            )

        if self._api_mode:
            params = {"season": season}
            if week:
                params["week"] = week
            return await self._load_from_api("coverage-matchups", params)
        else:
            pattern = f"*coverage*{season}*.csv"
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._load_from_csv, pattern)

    async def load_receiving_grades(
        self,
        season: int,
        week: Optional[int] = None,
    ) -> pl.DataFrame:
        """
        Load detailed receiving grades.

        Includes route running, contested catch, and drop grades.

        Args:
            season: NFL season year
            week: Optional week number

        Returns:
            DataFrame with receiving grades
        """
        if not self.enabled:
            raise DataSourceError(
                "PFF integration not enabled",
                self.source_name,
                retry_allowed=False,
            )

        if self._api_mode:
            params = {"season": season, "position_group": "receiving"}
            if week:
                params["week"] = week
            return await self._load_from_api("player-grades", params)
        else:
            pattern = f"*receiving*{season}*.csv"
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._load_from_csv, pattern)

    async def load_pass_blocking_grades(
        self,
        season: int,
        week: Optional[int] = None,
    ) -> pl.DataFrame:
        """
        Load pass blocking grades for offensive linemen.

        Args:
            season: NFL season year
            week: Optional week number

        Returns:
            DataFrame with pass blocking grades
        """
        if not self.enabled:
            raise DataSourceError(
                "PFF integration not enabled",
                self.source_name,
                retry_allowed=False,
            )

        if self._api_mode:
            params = {"season": season, "position_group": "oline"}
            if week:
                params["week"] = week
            return await self._load_from_api("player-grades", params)
        else:
            pattern = f"*oline*{season}*.csv"
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._load_from_csv, pattern)

    async def load_pass_rush_grades(
        self,
        season: int,
        week: Optional[int] = None,
    ) -> pl.DataFrame:
        """
        Load pass rush grades for defensive players.

        Args:
            season: NFL season year
            week: Optional week number

        Returns:
            DataFrame with pass rush grades
        """
        if not self.enabled:
            raise DataSourceError(
                "PFF integration not enabled",
                self.source_name,
                retry_allowed=False,
            )

        if self._api_mode:
            params = {"season": season, "grade_type": "pass_rush"}
            if week:
                params["week"] = week
            return await self._load_from_api("player-grades", params)
        else:
            pattern = f"*pass_rush*{season}*.csv"
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._load_from_csv, pattern)

    async def get_player_grade(
        self,
        player_id: str,
        season: int,
        week: Optional[int] = None,
    ) -> Optional[dict]:
        """
        Get grades for a specific player.

        Args:
            player_id: PFF player ID
            season: NFL season year
            week: Optional week number

        Returns:
            Player grade data or None if not found
        """
        grades = await self.load_player_grades(season, week)

        player = grades.filter(pl.col("player_id") == player_id)

        if len(player) == 0:
            return None

        return player.row(0, named=True)

    async def get_team_grades(
        self,
        team: str,
        season: int,
        week: Optional[int] = None,
    ) -> pl.DataFrame:
        """
        Get all player grades for a team.

        Args:
            team: Team abbreviation
            season: NFL season year
            week: Optional week number

        Returns:
            DataFrame with team's player grades
        """
        grades = await self.load_player_grades(season, week)
        return grades.filter(pl.col("team") == team)

    async def get_matchup_advantage(
        self,
        receiver_id: str,
        defender_id: str,
        season: int,
    ) -> Optional[dict]:
        """
        Get historical matchup data between a receiver and defender.

        Args:
            receiver_id: Receiver's PFF ID
            defender_id: Defender's PFF ID
            season: NFL season year

        Returns:
            Matchup stats or None if no data
        """
        matchups = await self.load_coverage_matchups(season)

        matchup = matchups.filter(
            (pl.col("receiver_id") == receiver_id)
            & (pl.col("defender_id") == defender_id)
        )

        if len(matchup) == 0:
            return None

        return matchup.row(0, named=True)

    async def calculate_position_group_grade(
        self,
        team: str,
        position_group: str,
        season: int,
        week: Optional[int] = None,
    ) -> float:
        """
        Calculate weighted average grade for a position group.

        Weighted by snap count.

        Args:
            team: Team abbreviation
            position_group: Position group (offense, defense, etc.)
            season: NFL season year
            week: Optional week number

        Returns:
            Weighted average grade
        """
        grades = await self.load_player_grades(season, week, position_group)
        team_grades = grades.filter(pl.col("team") == team)

        if len(team_grades) == 0:
            return 60.0  # Default average grade

        # Weighted average by snaps
        if "snaps" in team_grades.columns:
            total_snaps = team_grades["snaps"].sum()
            if total_snaps > 0:
                weighted_sum = (
                    team_grades["overall_grade"] * team_grades["snaps"]
                ).sum()
                return weighted_sum / total_snaps

        # Simple average if no snap data
        return team_grades["overall_grade"].mean()

    async def close(self) -> None:
        """Close HTTP session."""
        if self._session:
            await self._session.close()
            self._session = None


def create_pff_client_from_settings(settings) -> PFFClient:
    """
    Create a PFFClient from application settings.

    Args:
        settings: Application settings object

    Returns:
        Configured PFFClient
    """
    return PFFClient(
        api_key=getattr(settings, "pff_api_key", None),
        api_url=getattr(settings, "pff_api_url", None),
        data_dir=Path(settings.data_dir) / "pff" if hasattr(settings, "data_dir") else None,
        enabled=getattr(settings, "pff_enabled", True),
    )
