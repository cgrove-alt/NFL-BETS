"""
FTN Data / Football Outsiders DVOA integration.

Provides access to:
- DVOA (Defense-adjusted Value Over Average) team metrics
- Offensive and defensive efficiency ratings
- Weighted DVOA (recent games weighted more heavily)
- Opponent adjustments

DVOA measures a team's efficiency by comparing success on every play
to league average, adjusted for opponent strength.

Supports two modes:
- API mode: Direct API access (requires FTN subscription)
- CSV mode: Manual CSV upload from FTN Data exports
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


class FTNDVOAClient(CachedDataSource[pl.DataFrame]):
    """
    Client for accessing FTN/Football Outsiders DVOA metrics.

    DVOA (Defense-adjusted Value Over Average) is the premier
    efficiency metric for NFL teams. This client provides:

    - Total DVOA: Overall team efficiency
    - Offensive DVOA: Scoring efficiency
    - Defensive DVOA: Defensive efficiency (negative is better)
    - Special Teams DVOA: Kicking/return efficiency
    - Weighted DVOA: Recent performance weighted more heavily

    DVOA is expressed as a percentage:
    - 0%: League average
    - +10%: 10% better than average
    - -10%: 10% worse than average

    Top teams typically range from +20% to +30%.
    Bottom teams typically range from -20% to -30%.
    """

    # Expected columns from DVOA data
    TEAM_DVOA_COLUMNS = [
        "team",
        "total_dvoa",
        "total_dvoa_rank",
        "offense_dvoa",
        "offense_dvoa_rank",
        "defense_dvoa",
        "defense_dvoa_rank",
        "special_teams_dvoa",
        "special_teams_dvoa_rank",
        "weighted_dvoa",
        "weighted_dvoa_rank",
    ]

    SITUATIONAL_COLUMNS = [
        "team",
        "first_down_dvoa",
        "second_down_dvoa",
        "third_down_dvoa",
        "red_zone_dvoa",
        "late_close_dvoa",
    ]

    VARIANCE_COLUMNS = [
        "team",
        "variance",
        "variance_rank",
        "schedule_strength",
        "schedule_rank",
        "future_schedule",
        "future_schedule_rank",
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
            source_name="ftn_dvoa",
            cache_ttl_seconds=cache_ttl_hours * 3600,
            enabled=enabled,
        )

        self.api_key = api_key
        self.api_url = api_url or "https://api.ftndata.com/v1"
        self.data_dir = data_dir or Path("data/ftn")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Determine mode
        self._api_mode = bool(api_key)
        if self._api_mode:
            self.logger.info("FTN/DVOA client initialized in API mode")
        else:
            self.logger.info(
                f"FTN/DVOA client initialized in CSV mode. "
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
        raise NotImplementedError("Use specific methods like load_team_dvoa()")

    async def health_check(self) -> DataSourceHealth:
        """Check if DVOA data is available."""
        if not self.enabled:
            return DataSourceHealth(
                source_name=self.source_name,
                status=DataSourceStatus.DISABLED,
                error_message="FTN/DVOA integration disabled",
            )

        if self._api_mode:
            try:
                # Would make actual API health check
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
        """Load data from FTN API."""
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
        headers = {"X-API-Key": self.api_key}

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

        # Load most recent file matching pattern
        files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        latest_file = files[0]

        try:
            df = pl.read_csv(latest_file)
            self.logger.info(f"Loaded {len(df)} rows from {latest_file.name}")

            # Standardize column names
            df = self._standardize_columns(df)

            if required_columns:
                missing = set(required_columns) - set(df.columns)
                if missing:
                    self.logger.warning(f"Missing columns in DVOA data: {missing}")

            return df

        except Exception as e:
            raise DataSourceError(
                f"Failed to load {latest_file.name}: {e}",
                self.source_name,
                original_error=e,
                retry_allowed=False,
            )

    def _standardize_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """Standardize column names across different data formats."""
        # Common column name mappings
        column_map = {
            "Team": "team",
            "TEAM": "team",
            "Total DVOA": "total_dvoa",
            "TOTAL DVOA": "total_dvoa",
            "TOT DVOA": "total_dvoa",
            "Offense DVOA": "offense_dvoa",
            "OFF DVOA": "offense_dvoa",
            "OFF. DVOA": "offense_dvoa",
            "Defense DVOA": "defense_dvoa",
            "DEF DVOA": "defense_dvoa",
            "DEF. DVOA": "defense_dvoa",
            "Special Teams DVOA": "special_teams_dvoa",
            "ST DVOA": "special_teams_dvoa",
            "Weighted DVOA": "weighted_dvoa",
            "W.DVOA": "weighted_dvoa",
            "WDVOA": "weighted_dvoa",
            "Total Rank": "total_dvoa_rank",
            "Off Rank": "offense_dvoa_rank",
            "Def Rank": "defense_dvoa_rank",
            "ST Rank": "special_teams_dvoa_rank",
            "W.DVOA Rank": "weighted_dvoa_rank",
        }

        rename_dict = {}
        for col in df.columns:
            if col in column_map:
                rename_dict[col] = column_map[col]
            else:
                # Convert to snake_case
                new_name = col.lower().replace(" ", "_").replace(".", "")
                rename_dict[col] = new_name

        return df.rename(rename_dict)

    def _parse_dvoa_percentage(self, df: pl.DataFrame) -> pl.DataFrame:
        """Parse DVOA percentage strings to floats."""
        dvoa_columns = [
            "total_dvoa",
            "offense_dvoa",
            "defense_dvoa",
            "special_teams_dvoa",
            "weighted_dvoa",
        ]

        for col in dvoa_columns:
            if col in df.columns:
                # Handle percentage strings like "12.5%" or "-5.2%"
                if df[col].dtype == pl.Utf8:
                    df = df.with_columns(
                        pl.col(col)
                        .str.replace("%", "")
                        .str.strip_chars()
                        .cast(pl.Float64)
                        .alias(col)
                    )

        return df

    async def load_team_dvoa(
        self,
        season: int,
        week: Optional[int] = None,
    ) -> pl.DataFrame:
        """
        Load team DVOA metrics.

        Args:
            season: NFL season year
            week: Optional week number (None for latest)

        Returns:
            DataFrame with team DVOA metrics

        Example:
            >>> ftn = FTNDVOAClient(data_dir=Path("data/ftn"))
            >>> dvoa = await ftn.load_team_dvoa(2023)
            >>> top_offenses = dvoa.sort("offense_dvoa", descending=True).head(10)
        """
        if not self.enabled:
            raise DataSourceError(
                "FTN/DVOA integration not enabled",
                self.source_name,
                retry_allowed=False,
            )

        cache_path = self.data_dir / f"dvoa_{season}_{week or 'latest'}.parquet"

        # Check cache
        if cache_path.exists():
            cache_age = datetime.now().timestamp() - cache_path.stat().st_mtime
            if cache_age < self.cache_ttl_seconds:
                self.logger.debug(f"Loading DVOA from cache: {cache_path}")
                return pl.read_parquet(cache_path)

        if self._api_mode:
            params = {"season": season}
            if week:
                params["week"] = week
            df = await self._load_from_api("team-dvoa", params)
        else:
            if week:
                pattern = f"*dvoa*{season}*week{week}*.csv"
            else:
                pattern = f"*dvoa*{season}*.csv"

            loop = asyncio.get_event_loop()
            df = await loop.run_in_executor(None, self._load_from_csv, pattern)

        # Parse percentages
        df = self._parse_dvoa_percentage(df)

        # Cache the result
        df.write_parquet(cache_path)

        self._record_success(0)
        return df

    async def load_offensive_dvoa(
        self,
        season: int,
        week: Optional[int] = None,
    ) -> pl.DataFrame:
        """
        Load detailed offensive DVOA by category.

        Includes passing, rushing, and situational DVOA.

        Args:
            season: NFL season year
            week: Optional week number

        Returns:
            DataFrame with offensive DVOA breakdown
        """
        if not self.enabled:
            raise DataSourceError(
                "FTN/DVOA integration not enabled",
                self.source_name,
                retry_allowed=False,
            )

        if self._api_mode:
            params = {"season": season, "side": "offense"}
            if week:
                params["week"] = week
            return await self._load_from_api("dvoa-breakdown", params)
        else:
            pattern = f"*offense*dvoa*{season}*.csv"
            loop = asyncio.get_event_loop()
            df = await loop.run_in_executor(None, self._load_from_csv, pattern)
            return self._parse_dvoa_percentage(df)

    async def load_defensive_dvoa(
        self,
        season: int,
        week: Optional[int] = None,
    ) -> pl.DataFrame:
        """
        Load detailed defensive DVOA by category.

        Note: For defense, lower (more negative) is better.

        Args:
            season: NFL season year
            week: Optional week number

        Returns:
            DataFrame with defensive DVOA breakdown
        """
        if not self.enabled:
            raise DataSourceError(
                "FTN/DVOA integration not enabled",
                self.source_name,
                retry_allowed=False,
            )

        if self._api_mode:
            params = {"season": season, "side": "defense"}
            if week:
                params["week"] = week
            return await self._load_from_api("dvoa-breakdown", params)
        else:
            pattern = f"*defense*dvoa*{season}*.csv"
            loop = asyncio.get_event_loop()
            df = await loop.run_in_executor(None, self._load_from_csv, pattern)
            return self._parse_dvoa_percentage(df)

    async def load_situational_dvoa(
        self,
        season: int,
        week: Optional[int] = None,
    ) -> pl.DataFrame:
        """
        Load situational DVOA (red zone, third down, late & close).

        Args:
            season: NFL season year
            week: Optional week number

        Returns:
            DataFrame with situational DVOA
        """
        if not self.enabled:
            raise DataSourceError(
                "FTN/DVOA integration not enabled",
                self.source_name,
                retry_allowed=False,
            )

        if self._api_mode:
            params = {"season": season, "type": "situational"}
            if week:
                params["week"] = week
            return await self._load_from_api("dvoa-situational", params)
        else:
            pattern = f"*situational*{season}*.csv"
            loop = asyncio.get_event_loop()
            df = await loop.run_in_executor(None, self._load_from_csv, pattern)
            return self._parse_dvoa_percentage(df)

    async def load_variance_data(
        self,
        season: int,
    ) -> pl.DataFrame:
        """
        Load team variance and schedule strength data.

        Variance indicates how consistent a team performs.
        High variance teams are less predictable.

        Args:
            season: NFL season year

        Returns:
            DataFrame with variance and schedule data
        """
        if not self.enabled:
            raise DataSourceError(
                "FTN/DVOA integration not enabled",
                self.source_name,
                retry_allowed=False,
            )

        if self._api_mode:
            params = {"season": season}
            return await self._load_from_api("team-variance", params)
        else:
            pattern = f"*variance*{season}*.csv"
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._load_from_csv, pattern)

    async def get_team_dvoa(
        self,
        team: str,
        season: int,
        week: Optional[int] = None,
    ) -> Optional[dict]:
        """
        Get DVOA metrics for a specific team.

        Args:
            team: Team abbreviation
            season: NFL season year
            week: Optional week number

        Returns:
            Team DVOA data or None if not found
        """
        dvoa = await self.load_team_dvoa(season, week)

        team_data = dvoa.filter(pl.col("team") == team)

        if len(team_data) == 0:
            # Try with different team name formats
            team_data = dvoa.filter(
                pl.col("team").str.to_uppercase().str.contains(team.upper())
            )

        if len(team_data) == 0:
            return None

        return team_data.row(0, named=True)

    async def get_matchup_dvoa_differential(
        self,
        home_team: str,
        away_team: str,
        season: int,
        week: Optional[int] = None,
    ) -> dict:
        """
        Calculate DVOA differential for a matchup.

        Compares offensive DVOA vs opposing defensive DVOA.

        Args:
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            season: NFL season year
            week: Optional week number

        Returns:
            Dict with matchup differentials
        """
        dvoa = await self.load_team_dvoa(season, week)

        home_data = dvoa.filter(pl.col("team") == home_team)
        away_data = dvoa.filter(pl.col("team") == away_team)

        if len(home_data) == 0 or len(away_data) == 0:
            raise DataNotAvailableError(
                self.source_name,
                f"DVOA data not found for {home_team} or {away_team}",
            )

        home = home_data.row(0, named=True)
        away = away_data.row(0, named=True)

        # Calculate differentials
        # Positive means home team advantage
        return {
            "home_team": home_team,
            "away_team": away_team,
            # Home offense vs away defense
            "home_offense_vs_defense": (
                home.get("offense_dvoa", 0) - away.get("defense_dvoa", 0)
            ),
            # Away offense vs home defense
            "away_offense_vs_defense": (
                away.get("offense_dvoa", 0) - home.get("defense_dvoa", 0)
            ),
            # Total DVOA differential (home advantage)
            "total_dvoa_diff": home.get("total_dvoa", 0) - away.get("total_dvoa", 0),
            # Weighted DVOA differential (recent form)
            "weighted_dvoa_diff": (
                home.get("weighted_dvoa", 0) - away.get("weighted_dvoa", 0)
            ),
            # Raw team DVOAs
            "home_total_dvoa": home.get("total_dvoa", 0),
            "home_offense_dvoa": home.get("offense_dvoa", 0),
            "home_defense_dvoa": home.get("defense_dvoa", 0),
            "away_total_dvoa": away.get("total_dvoa", 0),
            "away_offense_dvoa": away.get("offense_dvoa", 0),
            "away_defense_dvoa": away.get("defense_dvoa", 0),
        }

    async def calculate_power_ranking(
        self,
        season: int,
        week: Optional[int] = None,
    ) -> pl.DataFrame:
        """
        Calculate power rankings based on weighted DVOA.

        Args:
            season: NFL season year
            week: Optional week number

        Returns:
            DataFrame with power rankings
        """
        dvoa = await self.load_team_dvoa(season, week)

        # Use weighted DVOA if available, otherwise total DVOA
        ranking_col = "weighted_dvoa" if "weighted_dvoa" in dvoa.columns else "total_dvoa"

        rankings = dvoa.select(["team", ranking_col]).sort(ranking_col, descending=True)

        rankings = rankings.with_columns(
            pl.lit(range(1, len(rankings) + 1)).alias("power_rank")
        )

        return rankings

    async def get_schedule_strength(
        self,
        team: str,
        season: int,
    ) -> Optional[dict]:
        """
        Get schedule strength data for a team.

        Args:
            team: Team abbreviation
            season: NFL season year

        Returns:
            Schedule strength data or None
        """
        try:
            variance = await self.load_variance_data(season)
            team_data = variance.filter(pl.col("team") == team)

            if len(team_data) == 0:
                return None

            return team_data.row(0, named=True)
        except DataNotAvailableError:
            return None

    async def close(self) -> None:
        """Close HTTP session."""
        if self._session:
            await self._session.close()
            self._session = None


def create_ftn_client_from_settings(settings) -> FTNDVOAClient:
    """
    Create an FTNDVOAClient from application settings.

    Args:
        settings: Application settings object

    Returns:
        Configured FTNDVOAClient
    """
    return FTNDVOAClient(
        api_key=getattr(settings, "ftn_api_key", None),
        api_url=getattr(settings, "ftn_api_url", None),
        data_dir=Path(settings.data_dir) / "ftn" if hasattr(settings, "data_dir") else None,
        enabled=getattr(settings, "ftn_enabled", True),
    )
