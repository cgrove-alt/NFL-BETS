"""
SIC Score (Sports Injury Central) integration for injury data.

Provides access to:
- Team health scores (0-100 scale)
- Player injury impact quantification
- Position group health assessments
- Historical injury data

SIC Score quantifies the expected performance impact of injuries
by considering player value, injury type, and recovery timeline.
"""
import asyncio
from datetime import datetime, timedelta
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


class SICScoreClient(CachedDataSource[pl.DataFrame]):
    """
    Client for accessing SIC Score injury data.

    SIC Score provides quantified injury impact metrics:
    - Team Health Score: 0-100 (100 = fully healthy)
    - Player Impact Score: Expected performance reduction
    - Position Group Health: Segment-specific health

    The service tracks:
    - Current injuries with expected return dates
    - Historical injury patterns
    - Recovery timelines
    - Position-weighted impact

    Supports two modes:
    - API mode: Real-time API access (subscription required)
    - CSV mode: Manual data import from SIC Score exports
    """

    # Position group definitions for health calculations
    POSITION_GROUPS = {
        "qb": ["QB"],
        "skill": ["RB", "WR", "TE", "FB"],
        "oline": ["LT", "LG", "C", "RG", "RT", "T", "G", "OL"],
        "dline": ["DE", "DT", "NT", "DL", "EDGE"],
        "linebacker": ["ILB", "OLB", "MLB", "LB"],
        "secondary": ["CB", "FS", "SS", "S", "DB"],
        "special_teams": ["K", "P", "LS"],
    }

    # Injury severity weights
    INJURY_SEVERITY = {
        "out": 1.0,  # Full impact
        "doubtful": 0.85,
        "questionable": 0.50,
        "probable": 0.15,
        "ir": 1.0,  # Injured reserve
        "pup": 1.0,  # Physically unable to perform
        "nfi": 1.0,  # Non-football injury
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        data_dir: Optional[Path] = None,
        cache_ttl_hours: int = 6,  # Injuries update frequently
        enabled: bool = True,
    ):
        super().__init__(
            source_name="sic_score",
            cache_ttl_seconds=cache_ttl_hours * 3600,
            enabled=enabled,
        )

        self.api_key = api_key
        self.api_url = api_url or "https://api.sicscore.com/v1"
        self.data_dir = data_dir or Path("data/sic")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Determine mode
        self._api_mode = bool(api_key)
        if self._api_mode:
            self.logger.info("SIC Score client initialized in API mode")
        else:
            self.logger.info(
                f"SIC Score client initialized in CSV mode. "
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
        raise NotImplementedError("Use specific methods like get_team_health_scores()")

    async def health_check(self) -> DataSourceHealth:
        """Check if SIC Score data is available."""
        if not self.enabled:
            return DataSourceHealth(
                source_name=self.source_name,
                status=DataSourceStatus.DISABLED,
                error_message="SIC Score integration disabled",
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
        """Load data from SIC Score API."""
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
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

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
                elif response.status == 429:
                    raise DataSourceError(
                        "Rate limit exceeded",
                        self.source_name,
                        retry_allowed=True,
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
    ) -> pl.DataFrame:
        """Load data from CSV files."""
        files = list(self.data_dir.glob(file_pattern))

        if not files:
            raise DataNotAvailableError(
                self.source_name,
                f"No files matching '{file_pattern}' found in {self.data_dir}",
            )

        # Load most recent file
        files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        latest_file = files[0]

        try:
            df = pl.read_csv(latest_file)
            self.logger.info(f"Loaded {len(df)} rows from {latest_file.name}")
            return self._standardize_columns(df)
        except Exception as e:
            raise DataSourceError(
                f"Failed to load {latest_file.name}: {e}",
                self.source_name,
                original_error=e,
                retry_allowed=False,
            )

    def _standardize_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """Standardize column names."""
        column_map = {
            "Team": "team",
            "TEAM": "team",
            "Player": "player_name",
            "PLAYER": "player_name",
            "Position": "position",
            "POS": "position",
            "Status": "status",
            "STATUS": "status",
            "Injury": "injury_type",
            "INJURY": "injury_type",
            "Impact": "impact_score",
            "IMPACT": "impact_score",
            "Health Score": "health_score",
            "HEALTH": "health_score",
            "Expected Return": "expected_return",
            "RETURN": "expected_return",
        }

        rename_dict = {}
        for col in df.columns:
            if col in column_map:
                rename_dict[col] = column_map[col]
            else:
                new_name = col.lower().replace(" ", "_").replace(".", "")
                rename_dict[col] = new_name

        return df.rename(rename_dict)

    async def get_team_health_scores(
        self,
        week: Optional[int] = None,
    ) -> pl.DataFrame:
        """
        Get health scores for all NFL teams.

        Args:
            week: Optional week number (None for current)

        Returns:
            DataFrame with team health scores

        Example:
            >>> sic = SICScoreClient()
            >>> health = await sic.get_team_health_scores()
            >>> healthiest = health.sort("health_score", descending=True).head(5)
        """
        if not self.enabled:
            raise DataSourceError(
                "SIC Score integration not enabled",
                self.source_name,
                retry_allowed=False,
            )

        cache_path = self.data_dir / f"team_health_{week or 'current'}.parquet"

        # Check cache
        if cache_path.exists():
            cache_age = datetime.now().timestamp() - cache_path.stat().st_mtime
            if cache_age < self.cache_ttl_seconds:
                self.logger.debug(f"Loading health scores from cache: {cache_path}")
                return pl.read_parquet(cache_path)

        if self._api_mode:
            params = {}
            if week:
                params["week"] = week
            df = await self._load_from_api("team-health", params)
        else:
            pattern = "*team*health*.csv"
            loop = asyncio.get_event_loop()
            df = await loop.run_in_executor(None, self._load_from_csv, pattern)

        # Cache the result
        df.write_parquet(cache_path)

        self._record_success(0)
        return df

    async def get_injury_report(
        self,
        team: Optional[str] = None,
    ) -> pl.DataFrame:
        """
        Get detailed injury report.

        Args:
            team: Optional team filter

        Returns:
            DataFrame with injury details
        """
        if not self.enabled:
            raise DataSourceError(
                "SIC Score integration not enabled",
                self.source_name,
                retry_allowed=False,
            )

        if self._api_mode:
            params = {}
            if team:
                params["team"] = team
            df = await self._load_from_api("injuries", params)
        else:
            pattern = "*injury*report*.csv"
            loop = asyncio.get_event_loop()
            df = await loop.run_in_executor(None, self._load_from_csv, pattern)

        if team and "team" in df.columns:
            df = df.filter(pl.col("team") == team)

        return df

    async def get_player_injury_impact(
        self,
        player_id: Optional[str] = None,
        player_name: Optional[str] = None,
    ) -> Optional[dict]:
        """
        Get injury impact for a specific player.

        Args:
            player_id: Player ID
            player_name: Player name (if ID not available)

        Returns:
            Player injury data or None
        """
        injuries = await self.get_injury_report()

        if player_id and "player_id" in injuries.columns:
            player = injuries.filter(pl.col("player_id") == player_id)
        elif player_name and "player_name" in injuries.columns:
            player = injuries.filter(
                pl.col("player_name").str.to_lowercase().str.contains(
                    player_name.lower()
                )
            )
        else:
            return None

        if len(player) == 0:
            return None

        return player.row(0, named=True)

    async def get_position_group_health(
        self,
        team: str,
        position_group: str,
    ) -> float:
        """
        Calculate health score for a position group.

        Args:
            team: Team abbreviation
            position_group: Position group (qb, skill, oline, etc.)

        Returns:
            Health score (0-100)
        """
        if position_group not in self.POSITION_GROUPS:
            raise ValueError(
                f"Invalid position group. Choose from: {list(self.POSITION_GROUPS.keys())}"
            )

        injuries = await self.get_injury_report(team)

        if len(injuries) == 0:
            return 100.0  # No injuries = full health

        positions = self.POSITION_GROUPS[position_group]

        # Filter to position group
        group_injuries = injuries.filter(pl.col("position").is_in(positions))

        if len(group_injuries) == 0:
            return 100.0

        # Calculate weighted impact
        total_impact = 0.0
        for row in group_injuries.iter_rows(named=True):
            status = row.get("status", "").lower()
            impact = row.get("impact_score", 0)

            # Weight by injury severity
            severity = self.INJURY_SEVERITY.get(status, 0.5)
            total_impact += impact * severity

        # Convert to health score (higher is better)
        # Assume max impact per player is ~10, max 3 key injuries per group
        max_expected_impact = 30.0
        health_score = max(0, 100 - (total_impact / max_expected_impact * 100))

        return round(health_score, 1)

    async def get_team_health(
        self,
        team: str,
    ) -> dict:
        """
        Get comprehensive health report for a team.

        Args:
            team: Team abbreviation

        Returns:
            Dict with overall and position group health
        """
        health_data = {
            "team": team,
            "overall_health": 0.0,
            "position_groups": {},
            "key_injuries": [],
        }

        # Get position group health
        total_health = 0.0
        for group in self.POSITION_GROUPS:
            group_health = await self.get_position_group_health(team, group)
            health_data["position_groups"][group] = group_health
            total_health += group_health

        # Overall is weighted average (QB weighted more heavily)
        weights = {
            "qb": 2.0,
            "skill": 1.5,
            "oline": 1.5,
            "dline": 1.0,
            "linebacker": 1.0,
            "secondary": 1.2,
            "special_teams": 0.5,
        }

        weighted_sum = sum(
            health_data["position_groups"].get(g, 100) * weights.get(g, 1.0)
            for g in self.POSITION_GROUPS
        )
        total_weight = sum(weights.values())
        health_data["overall_health"] = round(weighted_sum / total_weight, 1)

        # Get key injuries (impact > 5)
        try:
            injuries = await self.get_injury_report(team)
            if "impact_score" in injuries.columns:
                key_injuries = injuries.filter(pl.col("impact_score") > 5)
                health_data["key_injuries"] = key_injuries.to_dicts()
        except DataNotAvailableError:
            pass

        return health_data

    async def get_matchup_health_advantage(
        self,
        home_team: str,
        away_team: str,
    ) -> dict:
        """
        Calculate health advantage for a matchup.

        Args:
            home_team: Home team abbreviation
            away_team: Away team abbreviation

        Returns:
            Dict with health comparison
        """
        home_health = await self.get_team_health(home_team)
        away_health = await self.get_team_health(away_team)

        return {
            "home_team": home_team,
            "away_team": away_team,
            "home_health": home_health["overall_health"],
            "away_health": away_health["overall_health"],
            "health_advantage": (
                home_health["overall_health"] - away_health["overall_health"]
            ),
            "home_position_groups": home_health["position_groups"],
            "away_position_groups": away_health["position_groups"],
            # Specific matchup advantages
            "home_offense_vs_away_defense": (
                (
                    home_health["position_groups"].get("qb", 100)
                    + home_health["position_groups"].get("skill", 100)
                    + home_health["position_groups"].get("oline", 100)
                )
                / 3
                - (
                    away_health["position_groups"].get("dline", 100)
                    + away_health["position_groups"].get("linebacker", 100)
                    + away_health["position_groups"].get("secondary", 100)
                )
                / 3
            ),
            "away_offense_vs_home_defense": (
                (
                    away_health["position_groups"].get("qb", 100)
                    + away_health["position_groups"].get("skill", 100)
                    + away_health["position_groups"].get("oline", 100)
                )
                / 3
                - (
                    home_health["position_groups"].get("dline", 100)
                    + home_health["position_groups"].get("linebacker", 100)
                    + home_health["position_groups"].get("secondary", 100)
                )
                / 3
            ),
        }

    async def get_injury_history(
        self,
        player_id: Optional[str] = None,
        team: Optional[str] = None,
        seasons: int = 3,
    ) -> pl.DataFrame:
        """
        Get historical injury data.

        Args:
            player_id: Optional player filter
            team: Optional team filter
            seasons: Number of seasons to look back

        Returns:
            DataFrame with injury history
        """
        if not self.enabled:
            raise DataSourceError(
                "SIC Score integration not enabled",
                self.source_name,
                retry_allowed=False,
            )

        if self._api_mode:
            params = {"seasons": seasons}
            if player_id:
                params["player_id"] = player_id
            if team:
                params["team"] = team
            return await self._load_from_api("injury-history", params)
        else:
            pattern = "*injury*history*.csv"
            loop = asyncio.get_event_loop()
            df = await loop.run_in_executor(None, self._load_from_csv, pattern)

            if player_id and "player_id" in df.columns:
                df = df.filter(pl.col("player_id") == player_id)
            if team and "team" in df.columns:
                df = df.filter(pl.col("team") == team)

            return df

    async def predict_injury_risk(
        self,
        player_id: str,
    ) -> Optional[dict]:
        """
        Get injury risk prediction for a player.

        Based on historical injury patterns and current status.

        Args:
            player_id: Player ID

        Returns:
            Risk assessment or None
        """
        try:
            history = await self.get_injury_history(player_id=player_id)

            if len(history) == 0:
                return {
                    "player_id": player_id,
                    "risk_level": "low",
                    "risk_score": 0.1,
                    "injury_count": 0,
                    "message": "No significant injury history",
                }

            injury_count = len(history)
            recent_injuries = history.filter(
                pl.col("date") > (datetime.now() - timedelta(days=365)).isoformat()
            )

            # Simple risk calculation
            risk_score = min(1.0, (injury_count * 0.1) + (len(recent_injuries) * 0.2))

            if risk_score < 0.3:
                risk_level = "low"
            elif risk_score < 0.6:
                risk_level = "moderate"
            else:
                risk_level = "high"

            return {
                "player_id": player_id,
                "risk_level": risk_level,
                "risk_score": round(risk_score, 2),
                "injury_count": injury_count,
                "recent_injuries": len(recent_injuries),
            }

        except DataNotAvailableError:
            return None

    async def close(self) -> None:
        """Close HTTP session."""
        if self._session:
            await self._session.close()
            self._session = None


def create_sic_client_from_settings(settings) -> SICScoreClient:
    """
    Create a SICScoreClient from application settings.

    Args:
        settings: Application settings object

    Returns:
        Configured SICScoreClient
    """
    return SICScoreClient(
        api_key=getattr(settings, "sic_api_key", None),
        api_url=getattr(settings, "sic_api_url", None),
        data_dir=Path(settings.data_dir) / "sic" if hasattr(settings, "data_dir") else None,
        enabled=getattr(settings, "sic_enabled", True),
    )
