"""
The Odds API client for real-time sports betting odds.

Provides access to:
- Pre-game odds (spreads, moneylines, totals)
- Player prop odds
- Historical odds data
- Line movement tracking

Features async HTTP with rate limiting, credit tracking, and caching.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Any, Optional

import aiohttp
from loguru import logger

from .base import (
    CachedDataSource,
    DataSourceError,
    DataSourceHealth,
    DataSourceStatus,
    RateLimitError,
)


class OddsAPIClient(CachedDataSource[dict]):
    """
    Async client for The Odds API.

    The Odds API provides real-time odds from 40+ bookmakers for NFL and
    other sports. This client handles:
    - Async HTTP requests with connection pooling
    - Rate limiting to stay within API quotas
    - Credit tracking via response headers
    - Automatic caching with configurable TTL
    - Retry logic with exponential backoff

    Pricing tiers:
    - Free: 500 credits/month
    - Starter: $30/mo for 10,000 credits
    - Pro: $99/mo for 50,000 credits

    Credit costs:
    - Odds request: 1 credit per region per sport
    - Player props: 5 credits per event
    - Historical: 10 credits per request
    """

    BASE_URL = "https://api.the-odds-api.com/v4"
    SPORT_KEY = "americanfootball_nfl"

    # Default bookmakers to query
    DEFAULT_BOOKMAKERS = [
        "draftkings",
        "fanduel",
        "betmgm",
        "pointsbetus",
        "caesars",
        "bovada",
        "betonlineag",
    ]

    # Player prop types
    PLAYER_PROP_MARKETS = [
        "player_pass_yds",
        "player_pass_tds",
        "player_pass_attempts",
        "player_pass_completions",
        "player_rush_yds",
        "player_rush_attempts",
        "player_receptions",
        "player_reception_yds",
        "player_anytime_td",
        "player_1st_td",
    ]

    def __init__(
        self,
        api_key: str,
        cache_ttl_seconds: int = 300,
        enabled: bool = True,
        regions: list[str] | None = None,
        bookmakers: list[str] | None = None,
    ):
        super().__init__(
            source_name="odds_api",
            cache_ttl_seconds=cache_ttl_seconds,
            enabled=enabled,
        )

        self.api_key = api_key
        self.regions = regions or ["us", "us2"]
        self.bookmakers = bookmakers or self.DEFAULT_BOOKMAKERS

        # API credit tracking
        self._remaining_credits: int | None = None
        self._used_credits: int | None = None
        self._last_credit_check: datetime | None = None

        # Rate limiting
        self._request_semaphore = asyncio.Semaphore(5)  # Max concurrent requests
        self._last_request_time: datetime | None = None
        self._min_request_interval = 0.2  # 200ms between requests

        # Session management
        self._session: aiohttp.ClientSession | None = None

        if not api_key:
            self.logger.warning("No API key provided - odds API will be disabled")
            self.enabled = False

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create the aiohttp session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def _fetch_impl(self, *args, **kwargs) -> dict:
        """Implementation handled by specific methods."""
        raise NotImplementedError("Use specific methods like get_nfl_odds()")

    async def health_check(self) -> DataSourceHealth:
        """Check if The Odds API is accessible."""
        if not self.enabled:
            return DataSourceHealth(
                source_name=self.source_name,
                status=DataSourceStatus.DISABLED,
                error_message="API key not configured",
            )

        try:
            # Use sports endpoint for lightweight health check
            url = f"{self.BASE_URL}/sports"
            params = {"apiKey": self.api_key}

            session = await self._get_session()
            async with session.get(url, params=params) as response:
                self._update_credits(response.headers)

                if response.status == 200:
                    return DataSourceHealth(
                        source_name=self.source_name,
                        status=DataSourceStatus.HEALTHY,
                        last_success=datetime.now(),
                    )
                elif response.status == 401:
                    return DataSourceHealth(
                        source_name=self.source_name,
                        status=DataSourceStatus.UNHEALTHY,
                        error_message="Invalid API key",
                    )
                else:
                    return DataSourceHealth(
                        source_name=self.source_name,
                        status=DataSourceStatus.DEGRADED,
                        error_message=f"HTTP {response.status}",
                    )

        except Exception as e:
            return DataSourceHealth(
                source_name=self.source_name,
                status=DataSourceStatus.UNHEALTHY,
                error_message=str(e),
            )

    def _update_credits(self, headers: dict) -> None:
        """Update credit tracking from response headers."""
        if "x-requests-remaining" in headers:
            self._remaining_credits = int(headers["x-requests-remaining"])
        if "x-requests-used" in headers:
            self._used_credits = int(headers["x-requests-used"])
        self._last_credit_check = datetime.now()

        self.logger.debug(
            f"API credits - Remaining: {self._remaining_credits}, "
            f"Used: {self._used_credits}"
        )

        # Warn if running low
        if self._remaining_credits is not None and self._remaining_credits < 100:
            self.logger.warning(
                f"Low API credits! Only {self._remaining_credits} remaining"
            )

    async def _rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        if self._last_request_time:
            elapsed = (datetime.now() - self._last_request_time).total_seconds()
            if elapsed < self._min_request_interval:
                await asyncio.sleep(self._min_request_interval - elapsed)
        self._last_request_time = datetime.now()

    async def _make_request(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
    ) -> dict | list:
        """
        Make an authenticated request to The Odds API.

        Args:
            endpoint: API endpoint path
            params: Query parameters

        Returns:
            JSON response data

        Raises:
            DataSourceError: On API errors
            RateLimitError: When rate limited
        """
        if not self.enabled:
            raise DataSourceError(
                "Odds API not enabled", self.source_name, retry_allowed=False
            )

        url = f"{self.BASE_URL}/{endpoint}"
        request_params = {"apiKey": self.api_key}
        if params:
            request_params.update(params)

        async with self._request_semaphore:
            await self._rate_limit()

            session = await self._get_session()
            start_time = datetime.now()

            try:
                async with session.get(url, params=request_params) as response:
                    self._update_credits(response.headers)
                    elapsed = (datetime.now() - start_time).total_seconds()

                    if response.status == 200:
                        data = await response.json()
                        self._record_success(elapsed * 1000)
                        return data

                    elif response.status == 401:
                        raise DataSourceError(
                            "Invalid API key",
                            self.source_name,
                            retry_allowed=False,
                        )

                    elif response.status == 429:
                        retry_after = response.headers.get("Retry-After", "60")
                        raise RateLimitError(
                            self.source_name,
                            retry_after_seconds=int(retry_after),
                        )

                    elif response.status == 422:
                        error_text = await response.text()
                        raise DataSourceError(
                            f"Invalid request: {error_text}",
                            self.source_name,
                            retry_allowed=False,
                        )

                    else:
                        error_text = await response.text()
                        raise DataSourceError(
                            f"API error {response.status}: {error_text}",
                            self.source_name,
                            retry_allowed=True,
                        )

            except aiohttp.ClientError as e:
                self._record_failure(str(e))
                raise DataSourceError(
                    f"Connection error: {e}",
                    self.source_name,
                    original_error=e,
                    retry_allowed=True,
                )

    async def get_nfl_odds(
        self,
        markets: list[str] | None = None,
        bookmakers: list[str] | None = None,
        odds_format: str = "american",
    ) -> list[dict]:
        """
        Get current NFL odds for all upcoming games.

        Args:
            markets: Market types to fetch (h2h, spreads, totals)
            bookmakers: Specific bookmakers to query
            odds_format: Odds format (american, decimal)

        Returns:
            List of games with odds data

        Example:
            >>> client = OddsAPIClient(api_key="...")
            >>> odds = await client.get_nfl_odds(markets=["spreads", "totals"])
            >>> for game in odds:
            ...     print(f"{game['away_team']} @ {game['home_team']}")
        """
        markets = markets or ["h2h", "spreads", "totals"]
        bookmakers = bookmakers or self.bookmakers

        params = {
            "regions": ",".join(self.regions),
            "markets": ",".join(markets),
            "oddsFormat": odds_format,
            "bookmakers": ",".join(bookmakers),
        }

        self.logger.info(f"Fetching NFL odds for markets: {markets}")

        data = await self._make_request(f"sports/{self.SPORT_KEY}/odds", params)

        self.logger.info(f"Received odds for {len(data)} games")
        return data

    async def get_event_odds(
        self,
        event_id: str,
        markets: list[str] | None = None,
        bookmakers: list[str] | None = None,
        odds_format: str = "american",
    ) -> dict:
        """
        Get odds for a specific event/game.

        Args:
            event_id: The Odds API event ID
            markets: Market types to fetch
            bookmakers: Specific bookmakers to query
            odds_format: Odds format

        Returns:
            Event data with odds
        """
        markets = markets or ["h2h", "spreads", "totals"]
        bookmakers = bookmakers or self.bookmakers

        params = {
            "regions": ",".join(self.regions),
            "markets": ",".join(markets),
            "oddsFormat": odds_format,
            "bookmakers": ",".join(bookmakers),
        }

        self.logger.debug(f"Fetching odds for event: {event_id}")

        data = await self._make_request(
            f"sports/{self.SPORT_KEY}/events/{event_id}/odds", params
        )
        return data

    async def get_player_props(
        self,
        event_id: str,
        prop_markets: list[str] | None = None,
        bookmakers: list[str] | None = None,
        odds_format: str = "american",
    ) -> dict:
        """
        Get player prop odds for a specific event.

        Note: This costs 5 credits per request.

        Args:
            event_id: The Odds API event ID
            prop_markets: Prop types to fetch (player_pass_yds, etc.)
            bookmakers: Specific bookmakers to query
            odds_format: Odds format

        Returns:
            Event data with player prop odds

        Example:
            >>> props = await client.get_player_props(
            ...     event_id="abc123",
            ...     prop_markets=["player_pass_yds", "player_rush_yds"]
            ... )
        """
        prop_markets = prop_markets or self.PLAYER_PROP_MARKETS
        bookmakers = bookmakers or self.bookmakers

        params = {
            "regions": ",".join(self.regions),
            "markets": ",".join(prop_markets),
            "oddsFormat": odds_format,
            "bookmakers": ",".join(bookmakers),
        }

        self.logger.info(f"Fetching player props for event: {event_id}")

        data = await self._make_request(
            f"sports/{self.SPORT_KEY}/events/{event_id}/odds", params
        )
        return data

    async def get_events(self) -> list[dict]:
        """
        Get list of upcoming NFL events without odds.

        This is a lightweight call to get event IDs and matchup info.

        Returns:
            List of upcoming events
        """
        data = await self._make_request(f"sports/{self.SPORT_KEY}/events")
        self.logger.info(f"Found {len(data)} upcoming NFL events")
        return data

    async def get_all_player_props(
        self,
        prop_markets: list[str] | None = None,
        bookmakers: list[str] | None = None,
        max_events: int | None = None,
    ) -> list[dict]:
        """
        Get player props for all upcoming NFL games.

        Warning: This can use many API credits (5 per event).

        Args:
            prop_markets: Prop types to fetch
            bookmakers: Specific bookmakers
            max_events: Limit number of events to fetch

        Returns:
            List of events with player prop odds
        """
        events = await self.get_events()

        if max_events:
            events = events[:max_events]

        self.logger.info(f"Fetching player props for {len(events)} events")

        results = []
        for event in events:
            try:
                props = await self.get_player_props(
                    event_id=event["id"],
                    prop_markets=prop_markets,
                    bookmakers=bookmakers,
                )
                results.append(props)
            except DataSourceError as e:
                self.logger.warning(f"Failed to get props for {event['id']}: {e}")
                continue

        return results

    async def get_historical_odds(
        self,
        date: str,
        markets: list[str] | None = None,
    ) -> list[dict]:
        """
        Get historical odds for a specific date.

        Note: This costs 10 credits per request.

        Args:
            date: Date in ISO format (YYYY-MM-DDTHH:MM:SSZ)
            markets: Market types to fetch

        Returns:
            List of games with historical odds
        """
        markets = markets or ["h2h", "spreads", "totals"]

        params = {
            "regions": ",".join(self.regions),
            "markets": ",".join(markets),
            "date": date,
        }

        self.logger.info(f"Fetching historical odds for date: {date}")

        data = await self._make_request(
            f"historical/sports/{self.SPORT_KEY}/odds", params
        )
        return data

    def get_credit_status(self) -> dict:
        """
        Get current API credit status.

        Returns:
            Dict with remaining and used credits
        """
        return {
            "remaining": self._remaining_credits,
            "used": self._used_credits,
            "last_check": self._last_credit_check,
        }

    def parse_game_odds(self, game_data: dict) -> dict:
        """
        Parse game odds data into a structured format.

        Args:
            game_data: Raw game data from API

        Returns:
            Structured odds dictionary
        """
        result = {
            "id": game_data.get("id"),
            "commence_time": game_data.get("commence_time"),
            "home_team": game_data.get("home_team"),
            "away_team": game_data.get("away_team"),
            "bookmakers": {},
        }

        for bookmaker in game_data.get("bookmakers", []):
            book_key = bookmaker["key"]
            result["bookmakers"][book_key] = {
                "title": bookmaker.get("title"),
                "last_update": bookmaker.get("last_update"),
                "markets": {},
            }

            for market in bookmaker.get("markets", []):
                market_key = market["key"]
                outcomes = {}

                for outcome in market.get("outcomes", []):
                    outcomes[outcome["name"]] = {
                        "price": outcome.get("price"),
                        "point": outcome.get("point"),
                    }

                result["bookmakers"][book_key]["markets"][market_key] = outcomes

        return result

    def find_best_odds(
        self,
        game_data: dict,
        market: str,
        outcome: str,
    ) -> tuple[str, int, float | None] | None:
        """
        Find the best odds across bookmakers for a specific outcome.

        Args:
            game_data: Parsed game data
            market: Market type (spreads, h2h, totals)
            outcome: Outcome name (team name or Over/Under)

        Returns:
            Tuple of (bookmaker, odds, point) or None if not found
        """
        best_book = None
        best_odds = float("-inf")
        best_point = None

        for book_key, book_data in game_data.get("bookmakers", {}).items():
            markets = book_data.get("markets", {})
            if market not in markets:
                continue

            outcomes = markets[market]
            if outcome not in outcomes:
                continue

            odds = outcomes[outcome].get("price", 0)
            point = outcomes[outcome].get("point")

            if odds > best_odds:
                best_odds = odds
                best_book = book_key
                best_point = point

        if best_book is None:
            return None

        return (best_book, int(best_odds), best_point)

    async def get_spread_comparison(self) -> list[dict]:
        """
        Get spread comparison across all bookmakers for arbitrage detection.

        Returns:
            List of games with spread data from all books
        """
        games = await self.get_nfl_odds(markets=["spreads"])

        comparisons = []
        for game in games:
            parsed = self.parse_game_odds(game)

            comparison = {
                "id": parsed["id"],
                "home_team": parsed["home_team"],
                "away_team": parsed["away_team"],
                "commence_time": parsed["commence_time"],
                "spreads": [],
            }

            for book_key, book_data in parsed["bookmakers"].items():
                spreads = book_data.get("markets", {}).get("spreads", {})
                if spreads:
                    for team, data in spreads.items():
                        comparison["spreads"].append(
                            {
                                "bookmaker": book_key,
                                "team": team,
                                "point": data.get("point"),
                                "price": data.get("price"),
                            }
                        )

            comparisons.append(comparison)

        return comparisons

    async def monitor_line_movement(
        self,
        event_id: str,
        interval_seconds: int = 300,
        duration_minutes: int = 60,
    ) -> list[dict]:
        """
        Monitor line movement for a specific event.

        Args:
            event_id: Event to monitor
            interval_seconds: Polling interval
            duration_minutes: How long to monitor

        Yields:
            Snapshots of odds over time
        """
        snapshots = []
        end_time = datetime.now() + timedelta(minutes=duration_minutes)

        while datetime.now() < end_time:
            try:
                odds = await self.get_event_odds(event_id, markets=["spreads", "totals"])
                snapshot = {
                    "timestamp": datetime.now().isoformat(),
                    "odds": self.parse_game_odds(odds),
                }
                snapshots.append(snapshot)
                self.logger.debug(f"Captured odds snapshot for {event_id}")
            except DataSourceError as e:
                self.logger.warning(f"Failed to capture snapshot: {e}")

            await asyncio.sleep(interval_seconds)

        return snapshots


class OddsAPIClientFactory:
    """Factory for creating OddsAPIClient instances."""

    @staticmethod
    def create_from_settings(settings) -> OddsAPIClient:
        """
        Create an OddsAPIClient from application settings.

        Args:
            settings: Application settings object

        Returns:
            Configured OddsAPIClient
        """
        return OddsAPIClient(
            api_key=settings.odds_api.api_key or "",
            cache_ttl_seconds=settings.odds_api.cache_ttl_seconds,
            enabled=bool(settings.odds_api.api_key),
            regions=settings.odds_api.regions,
            bookmakers=settings.odds_api.default_bookmakers,
        )
