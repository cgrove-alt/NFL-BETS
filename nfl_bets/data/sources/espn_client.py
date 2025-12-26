"""
ESPN API client for real-time roster and injury data.

ESPN provides hidden/undocumented APIs that are free and require no authentication.
These are the same endpoints used by ESPN's website and mobile apps.

Data provided:
- Current team rosters with player status
- Real-time injury reports (Questionable/Doubtful/Out)
- Depth chart positions

Note: These are unofficial APIs and may change without notice.
"""
import asyncio
from datetime import datetime
from typing import Any, Optional

import aiohttp
import polars as pl
from loguru import logger

from .base import (
    CachedDataSource,
    DataSourceError,
    DataSourceHealth,
    DataSourceStatus,
    DataNotAvailableError,
)


class ESPNClient(CachedDataSource[pl.DataFrame]):
    """
    Client for accessing ESPN's hidden NFL APIs.

    Provides real-time access to:
    - Team rosters with player status
    - Injury reports with game status
    - Depth chart positions

    No API key required - these are public endpoints.
    """

    # Base URLs for ESPN APIs
    SITE_API = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"
    CORE_API = "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl"

    # ESPN team ID mapping (NFL team abbreviation -> ESPN team ID)
    TEAM_IDS = {
        "ARI": 22,
        "ATL": 1,
        "BAL": 33,
        "BUF": 2,
        "CAR": 29,
        "CHI": 3,
        "CIN": 4,
        "CLE": 5,
        "DAL": 6,
        "DEN": 7,
        "DET": 8,
        "GB": 9,
        "HOU": 34,
        "IND": 11,
        "JAX": 30,
        "KC": 12,
        "LAC": 24,
        "LAR": 14,
        "LV": 13,
        "MIA": 15,
        "MIN": 16,
        "NE": 17,
        "NO": 18,
        "NYG": 19,
        "NYJ": 20,
        "PHI": 21,
        "PIT": 23,
        "SEA": 26,
        "SF": 25,
        "TB": 27,
        "TEN": 10,
        "WAS": 28,
    }

    # Reverse mapping for lookups
    ID_TO_TEAM = {v: k for k, v in TEAM_IDS.items()}

    # Position mapping for depth charts
    SKILL_POSITIONS = ["QB", "RB", "WR", "TE", "FB"]

    def __init__(
        self,
        cache_ttl_seconds: int = 1800,  # 30 minutes default
        enabled: bool = True,
    ):
        super().__init__(
            source_name="espn",
            cache_ttl_seconds=cache_ttl_seconds,
            enabled=enabled,
        )
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            # Create connector with SSL context that handles cert issues
            import ssl
            import certifi

            try:
                # Try using certifi certificates
                ssl_context = ssl.create_default_context(cafile=certifi.where())
                connector = aiohttp.TCPConnector(ssl=ssl_context)
            except Exception:
                # Fallback: disable SSL verification (for local dev only)
                connector = aiohttp.TCPConnector(ssl=False)
                self.logger.warning("Using SSL-disabled connector for ESPN API")

            self._session = aiohttp.ClientSession(
                connector=connector,
                headers={"User-Agent": "Mozilla/5.0 (compatible; NFL-Bets/1.0)"}
            )
        return self._session

    async def _fetch_impl(self, *args, **kwargs) -> pl.DataFrame:
        """Implementation handled by specific methods."""
        raise NotImplementedError("Use specific methods like get_team_roster()")

    async def health_check(self) -> DataSourceHealth:
        """Check if ESPN API is available."""
        if not self.enabled:
            return DataSourceHealth(
                source_name=self.source_name,
                status=DataSourceStatus.DISABLED,
                error_message="ESPN integration disabled",
            )

        try:
            # Quick check - fetch KC roster (always available)
            session = await self._get_session()
            url = f"{self.SITE_API}/teams/12/roster"

            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    return DataSourceHealth(
                        source_name=self.source_name,
                        status=DataSourceStatus.HEALTHY,
                        last_success=datetime.now(),
                    )
                else:
                    return DataSourceHealth(
                        source_name=self.source_name,
                        status=DataSourceStatus.UNHEALTHY,
                        error_message=f"HTTP {response.status}",
                    )

        except Exception as e:
            return DataSourceHealth(
                source_name=self.source_name,
                status=DataSourceStatus.UNHEALTHY,
                error_message=str(e),
            )

    def _get_team_id(self, team: str) -> int:
        """Convert team abbreviation to ESPN team ID."""
        team_upper = team.upper()
        if team_upper not in self.TEAM_IDS:
            raise DataSourceError(
                f"Unknown team: {team}",
                self.source_name,
                retry_allowed=False,
            )
        return self.TEAM_IDS[team_upper]

    async def get_team_roster(
        self,
        team: str,
        position: Optional[str] = None,
    ) -> pl.DataFrame:
        """
        Get current roster for a team.

        Args:
            team: Team abbreviation (e.g., "KC", "BUF")
            position: Optional position filter (e.g., "QB", "RB")

        Returns:
            DataFrame with columns:
            - player_name, player_id, position, jersey_number
            - status (Active, Injured, Practice Squad, etc.)
            - experience, college, height, weight
        """
        if not self.enabled:
            return pl.DataFrame()

        team_id = self._get_team_id(team)
        session = await self._get_session()
        url = f"{self.SITE_API}/teams/{team_id}/roster"

        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status != 200:
                    raise DataSourceError(
                        f"ESPN API returned {response.status}",
                        self.source_name,
                        retry_allowed=True,
                    )

                data = await response.json()

        except aiohttp.ClientError as e:
            raise DataSourceError(
                f"Connection error: {e}",
                self.source_name,
                original_error=e,
                retry_allowed=True,
            )

        # Parse roster response
        players = []
        for group in data.get("athletes", []):
            for athlete in group.get("items", []):
                player = {
                    "player_id": str(athlete.get("id", "")),
                    "player_name": athlete.get("fullName", ""),
                    "first_name": athlete.get("firstName", ""),
                    "last_name": athlete.get("lastName", ""),
                    "position": athlete.get("position", {}).get("abbreviation", ""),
                    "jersey_number": athlete.get("jersey", ""),
                    "status": athlete.get("status", {}).get("name", "Active"),
                    "experience": athlete.get("experience", {}).get("years", 0),
                    "college": athlete.get("college", {}).get("name", ""),
                    "height": athlete.get("height", 0),
                    "weight": athlete.get("weight", 0),
                    "team": team.upper(),
                }
                players.append(player)

        df = pl.DataFrame(players)

        # Filter by position if specified
        if position and len(df) > 0:
            df = df.filter(pl.col("position") == position.upper())

        self._record_success(0)
        return df

    async def get_team_injuries(
        self,
        team: str,
    ) -> pl.DataFrame:
        """
        Get current injury report for a team.

        Args:
            team: Team abbreviation (e.g., "KC", "BUF")

        Returns:
            DataFrame with columns:
            - player_name, player_id, position
            - status (Out, Doubtful, Questionable, Probable)
            - injury_type, injury_detail
        """
        if not self.enabled:
            return pl.DataFrame()

        team_id = self._get_team_id(team)
        session = await self._get_session()
        url = f"{self.CORE_API}/teams/{team_id}/injuries"

        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status == 404:
                    # No injuries reported
                    return pl.DataFrame()

                if response.status != 200:
                    raise DataSourceError(
                        f"ESPN API returned {response.status}",
                        self.source_name,
                        retry_allowed=True,
                    )

                data = await response.json()

        except aiohttp.ClientError as e:
            raise DataSourceError(
                f"Connection error: {e}",
                self.source_name,
                original_error=e,
                retry_allowed=True,
            )

        # Parse injuries - ESPN returns list of injury references
        injuries = []
        items = data.get("items", [])

        for item in items:
            # Each item has a $ref URL we need to fetch
            ref_url = item.get("$ref")
            if ref_url:
                try:
                    async with session.get(ref_url, timeout=aiohttp.ClientTimeout(total=10)) as ref_response:
                        if ref_response.status == 200:
                            injury_data = await ref_response.json()
                            athlete = injury_data.get("athlete", {})
                            injuries.append({
                                "player_id": str(athlete.get("id", "")),
                                "player_name": athlete.get("fullName", ""),
                                "position": athlete.get("position", {}).get("abbreviation", ""),
                                "status": injury_data.get("status", "Unknown"),
                                "injury_type": injury_data.get("type", {}).get("text", ""),
                                "injury_detail": injury_data.get("details", {}).get("detail", ""),
                                "team": team.upper(),
                            })
                except Exception as e:
                    self.logger.debug(f"Failed to fetch injury detail: {e}")
                    continue

        df = pl.DataFrame(injuries) if injuries else pl.DataFrame()
        self._record_success(0)
        return df

    async def get_depth_chart(
        self,
        team: str,
        season: int = 2024,
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Get depth chart for a team.

        Args:
            team: Team abbreviation (e.g., "KC", "BUF")
            season: NFL season year

        Returns:
            Dict mapping position to list of players in depth order:
            {
                "QB": [{"name": "Patrick Mahomes", "depth": 1}, ...],
                "RB": [...],
                ...
            }
        """
        if not self.enabled:
            return {}

        team_id = self._get_team_id(team)
        session = await self._get_session()
        url = f"{self.CORE_API}/seasons/{season}/teams/{team_id}/depthcharts"

        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status == 404:
                    return {}

                if response.status != 200:
                    raise DataSourceError(
                        f"ESPN API returned {response.status}",
                        self.source_name,
                        retry_allowed=True,
                    )

                data = await response.json()

        except aiohttp.ClientError as e:
            raise DataSourceError(
                f"Connection error: {e}",
                self.source_name,
                original_error=e,
                retry_allowed=True,
            )

        # Parse depth chart
        depth_chart: dict[str, list[dict[str, Any]]] = {}

        for item in data.get("items", []):
            positions = item.get("positions", {})
            for pos_key, pos_data in positions.items():
                athletes = pos_data.get("athletes", [])
                if athletes:
                    depth_chart[pos_key.upper()] = [
                        {
                            "name": a.get("athlete", {}).get("fullName", ""),
                            "player_id": str(a.get("athlete", {}).get("id", "")),
                            "depth": a.get("slot", 1),
                        }
                        for a in sorted(athletes, key=lambda x: x.get("slot", 1))
                    ]

        self._record_success(0)
        return depth_chart

    async def get_key_players(
        self,
        team: str,
        season: int = 2024,
        positions: Optional[list[str]] = None,
        max_per_position: int = 2,
        exclude_injured: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Get key skill players for a team (QB, RB, WR, TE).

        Uses roster data and injuries to return top players.
        Players are ordered by experience/jersey number as a proxy for starter.

        Args:
            team: Team abbreviation
            season: NFL season year
            positions: List of positions to include (default: QB, RB, WR, TE)
            max_per_position: Max players per position
            exclude_injured: Whether to exclude injured players

        Returns:
            List of player dicts with name, position, status
        """
        if positions is None:
            positions = self.SKILL_POSITIONS

        # Fetch roster and injuries in parallel
        roster_task = self.get_team_roster(team)
        injuries_task = self.get_team_injuries(team)

        roster, injuries = await asyncio.gather(
            roster_task, injuries_task,
            return_exceptions=True
        )

        # Handle errors gracefully
        if isinstance(roster, Exception):
            self.logger.warning(f"Failed to get roster for {team}: {roster}")
            return []
        if isinstance(injuries, Exception):
            self.logger.warning(f"Failed to get injuries for {team}: {injuries}")
            injuries = pl.DataFrame()

        if len(roster) == 0:
            return []

        # Build injury status map
        injury_status_map = {}
        if len(injuries) > 0:
            for row in injuries.iter_rows(named=True):
                pid = row.get("player_id", "")
                status = row.get("status", "").upper()
                injury_status_map[pid] = status

        # Get players by position from roster
        key_players = []

        for pos in positions:
            pos_upper = pos.upper()

            # Filter roster by position
            pos_players = roster.filter(pl.col("position") == pos_upper)

            if len(pos_players) == 0:
                continue

            # Sort by experience (desc) as proxy for starter status
            # More experienced players are typically starters
            pos_players = pos_players.sort("experience", descending=True)

            count = 0
            for row in pos_players.iter_rows(named=True):
                if count >= max_per_position:
                    break

                pid = row.get("player_id", "")
                name = row.get("player_name", "")
                status = injury_status_map.get(pid, "ACTIVE")

                # Skip if on IR/PUP (not on active roster)
                roster_status = row.get("status", "")
                if roster_status in ("Injured Reserve", "Physically Unable to Perform"):
                    status = "IR"

                # Skip injured players if requested
                if exclude_injured and status in ("OUT", "IR", "DOUBTFUL"):
                    continue

                key_players.append({
                    "name": name,
                    "player_id": pid,
                    "position": pos_upper,
                    "team": team.upper(),
                    "injury_status": status,
                    "experience": row.get("experience", 0),
                })
                count += 1

        return key_players

    async def get_player_status(
        self,
        player_name: str,
        team: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        """
        Get current status for a specific player.

        Args:
            player_name: Player's full name
            team: Optional team abbreviation to narrow search

        Returns:
            Player status dict or None if not found
        """
        teams_to_search = [team] if team else list(self.TEAM_IDS.keys())
        name_lower = player_name.lower()

        for t in teams_to_search:
            try:
                roster = await self.get_team_roster(t)
                if len(roster) == 0:
                    continue

                # Search for player
                matches = roster.filter(
                    pl.col("player_name").str.to_lowercase().str.contains(name_lower)
                )

                if len(matches) > 0:
                    player = matches.row(0, named=True)

                    # Check injury status
                    injuries = await self.get_team_injuries(t)
                    injury_status = "ACTIVE"
                    injury_detail = None

                    if len(injuries) > 0:
                        player_injuries = injuries.filter(
                            pl.col("player_id") == player.get("player_id")
                        )
                        if len(player_injuries) > 0:
                            injury = player_injuries.row(0, named=True)
                            injury_status = injury.get("status", "ACTIVE")
                            injury_detail = injury.get("injury_type")

                    return {
                        "player_id": player.get("player_id"),
                        "player_name": player.get("player_name"),
                        "team": t,
                        "position": player.get("position"),
                        "jersey_number": player.get("jersey_number"),
                        "status": injury_status,
                        "injury_detail": injury_detail,
                    }

            except Exception as e:
                self.logger.debug(f"Error searching {t}: {e}")
                continue

        return None

    async def close(self) -> None:
        """Close HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
