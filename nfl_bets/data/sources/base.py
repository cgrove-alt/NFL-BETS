"""
Abstract base class for all data source clients.

Provides common interface, error handling, logging, and health check patterns
that all data source implementations must follow.
"""
import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Generic, Optional, TypeVar

from loguru import logger

T = TypeVar("T")


class DataSourceStatus(str, Enum):
    """Health status of a data source."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    DISABLED = "disabled"


@dataclass
class DataSourceHealth:
    """Health information for a data source."""

    source_name: str
    status: DataSourceStatus
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    consecutive_failures: int = 0
    error_message: Optional[str] = None
    latency_ms: Optional[float] = None


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_attempts: int = 3
    initial_delay_seconds: float = 1.0
    max_delay_seconds: float = 30.0
    exponential_base: float = 2.0
    jitter: bool = True


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker pattern."""

    failure_threshold: int = 5
    recovery_timeout_seconds: int = 60
    half_open_max_calls: int = 3


@dataclass
class CircuitBreakerState:
    """State tracking for circuit breaker."""

    failures: int = 0
    last_failure_time: Optional[datetime] = None
    state: str = "closed"  # closed, open, half-open
    half_open_successes: int = 0


class DataSourceError(Exception):
    """Base exception for data source errors."""

    def __init__(
        self,
        message: str,
        source_name: str,
        original_error: Optional[Exception] = None,
        retry_allowed: bool = True,
    ):
        super().__init__(message)
        self.source_name = source_name
        self.original_error = original_error
        self.retry_allowed = retry_allowed


class RateLimitError(DataSourceError):
    """Error when rate limit is exceeded."""

    def __init__(
        self,
        source_name: str,
        retry_after_seconds: Optional[int] = None,
    ):
        super().__init__(
            f"Rate limit exceeded for {source_name}",
            source_name,
            retry_allowed=True,
        )
        self.retry_after_seconds = retry_after_seconds


class AuthenticationError(DataSourceError):
    """Error when authentication fails."""

    def __init__(self, source_name: str, message: str = "Authentication failed"):
        super().__init__(message, source_name, retry_allowed=False)


class DataNotAvailableError(DataSourceError):
    """Error when requested data is not available."""

    def __init__(self, source_name: str, message: str):
        super().__init__(message, source_name, retry_allowed=False)


class BaseDataSource(ABC, Generic[T]):
    """
    Abstract base class for all data sources.

    Provides:
    - Common interface for fetching data
    - Retry logic with exponential backoff
    - Circuit breaker pattern
    - Health monitoring
    - Logging
    - Caching interface

    All data source implementations should inherit from this class.
    """

    def __init__(
        self,
        source_name: str,
        enabled: bool = True,
        retry_config: Optional[RetryConfig] = None,
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
    ):
        self.source_name = source_name
        self.enabled = enabled
        self.retry_config = retry_config or RetryConfig()
        self.circuit_breaker_config = circuit_breaker_config or CircuitBreakerConfig()

        # State tracking
        self._circuit_breaker = CircuitBreakerState()
        self._health = DataSourceHealth(
            source_name=source_name,
            status=DataSourceStatus.HEALTHY if enabled else DataSourceStatus.DISABLED,
        )

        # Setup logging
        self.logger = logger.bind(source=source_name)

    @property
    def is_available(self) -> bool:
        """Check if the data source is available for use."""
        if not self.enabled:
            return False
        if self._circuit_breaker.state == "open":
            # Check if recovery timeout has passed
            if self._circuit_breaker.last_failure_time:
                elapsed = datetime.now() - self._circuit_breaker.last_failure_time
                if elapsed.total_seconds() >= self.circuit_breaker_config.recovery_timeout_seconds:
                    self._circuit_breaker.state = "half-open"
                    self._circuit_breaker.half_open_successes = 0
                    self.logger.info("Circuit breaker entering half-open state")
                    return True
            return False
        return True

    @abstractmethod
    async def _fetch_impl(self, *args, **kwargs) -> T:
        """
        Implementation of the actual fetch logic.

        Subclasses must implement this method.
        Should not include retry logic - that's handled by fetch().
        """
        pass

    @abstractmethod
    async def health_check(self) -> DataSourceHealth:
        """
        Perform a health check on the data source.

        Should be a lightweight check (e.g., ping endpoint).
        """
        pass

    async def fetch(self, *args, **kwargs) -> T:
        """
        Fetch data with retry logic and circuit breaker.

        This is the main entry point for data retrieval.
        Handles retries, circuit breaker, and error logging.
        """
        if not self.is_available:
            raise DataSourceError(
                f"Data source {self.source_name} is not available",
                self.source_name,
                retry_allowed=False,
            )

        start_time = datetime.now()
        last_error: Optional[Exception] = None

        for attempt in range(1, self.retry_config.max_attempts + 1):
            try:
                self.logger.debug(f"Fetch attempt {attempt}/{self.retry_config.max_attempts}")

                result = await self._fetch_impl(*args, **kwargs)

                # Success - update state
                elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000
                self._record_success(elapsed_ms)

                return result

            except DataSourceError as e:
                last_error = e
                self.logger.warning(
                    f"Fetch error on attempt {attempt}: {e}",
                    error=str(e),
                    attempt=attempt,
                )

                if not e.retry_allowed:
                    self._record_failure(str(e))
                    raise

                if attempt < self.retry_config.max_attempts:
                    delay = self._calculate_delay(attempt)
                    self.logger.info(f"Retrying in {delay:.2f} seconds...")
                    await asyncio.sleep(delay)

            except Exception as e:
                last_error = e
                self.logger.error(
                    f"Unexpected error on attempt {attempt}: {e}",
                    error=str(e),
                    attempt=attempt,
                    exc_info=True,
                )

                if attempt < self.retry_config.max_attempts:
                    delay = self._calculate_delay(attempt)
                    await asyncio.sleep(delay)

        # All retries exhausted
        self._record_failure(str(last_error) if last_error else "Unknown error")
        raise DataSourceError(
            f"All {self.retry_config.max_attempts} attempts failed for {self.source_name}",
            self.source_name,
            original_error=last_error,
            retry_allowed=False,
        )

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay before next retry using exponential backoff."""
        delay = self.retry_config.initial_delay_seconds * (
            self.retry_config.exponential_base ** (attempt - 1)
        )
        delay = min(delay, self.retry_config.max_delay_seconds)

        if self.retry_config.jitter:
            import random
            delay = delay * (0.5 + random.random())

        return delay

    def _record_success(self, latency_ms: float) -> None:
        """Record a successful fetch."""
        self._health.last_success = datetime.now()
        self._health.latency_ms = latency_ms
        self._health.consecutive_failures = 0
        self._health.error_message = None
        self._health.status = DataSourceStatus.HEALTHY

        # Circuit breaker handling
        if self._circuit_breaker.state == "half-open":
            self._circuit_breaker.half_open_successes += 1
            if (
                self._circuit_breaker.half_open_successes
                >= self.circuit_breaker_config.half_open_max_calls
            ):
                self._circuit_breaker.state = "closed"
                self._circuit_breaker.failures = 0
                self.logger.info("Circuit breaker closed after successful recovery")
        else:
            self._circuit_breaker.failures = 0

    def _record_failure(self, error_message: str) -> None:
        """Record a failed fetch."""
        self._health.last_failure = datetime.now()
        self._health.consecutive_failures += 1
        self._health.error_message = error_message

        # Update circuit breaker
        self._circuit_breaker.failures += 1
        self._circuit_breaker.last_failure_time = datetime.now()

        # Check if we should open the circuit
        if self._circuit_breaker.failures >= self.circuit_breaker_config.failure_threshold:
            self._circuit_breaker.state = "open"
            self._health.status = DataSourceStatus.UNHEALTHY
            self.logger.error(
                f"Circuit breaker opened after {self._circuit_breaker.failures} failures"
            )
        elif self._health.consecutive_failures >= 2:
            self._health.status = DataSourceStatus.DEGRADED

    def get_health(self) -> DataSourceHealth:
        """Get current health status of the data source."""
        return self._health

    def reset_circuit_breaker(self) -> None:
        """Manually reset the circuit breaker."""
        self._circuit_breaker = CircuitBreakerState()
        self._health.status = DataSourceStatus.HEALTHY
        self._health.consecutive_failures = 0
        self._health.error_message = None
        self.logger.info("Circuit breaker manually reset")


class CachedDataSource(BaseDataSource[T]):
    """
    Data source with built-in caching support.

    Extends BaseDataSource to add caching functionality.
    Uses the cache manager to store and retrieve data.
    """

    def __init__(
        self,
        source_name: str,
        cache_ttl_seconds: int = 300,
        cache_prefix: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(source_name, **kwargs)
        self.cache_ttl_seconds = cache_ttl_seconds
        self.cache_prefix = cache_prefix or source_name
        self._cache = None  # Will be set by set_cache()

    def set_cache(self, cache) -> None:
        """Set the cache manager instance."""
        self._cache = cache

    def _get_cache_key(self, *args, **kwargs) -> str:
        """
        Generate a cache key from the fetch arguments.

        Override in subclasses for custom key generation.
        """
        import hashlib
        import json

        key_data = {
            "args": args,
            "kwargs": {k: v for k, v in sorted(kwargs.items())},
        }
        key_hash = hashlib.md5(json.dumps(key_data, default=str).encode()).hexdigest()
        return f"{self.cache_prefix}:{key_hash}"

    async def fetch_cached(self, *args, use_cache: bool = True, **kwargs) -> T:
        """
        Fetch data with caching.

        Args:
            *args: Arguments to pass to fetch
            use_cache: Whether to use cached data if available
            **kwargs: Keyword arguments to pass to fetch

        Returns:
            The fetched data
        """
        if not use_cache or self._cache is None:
            return await self.fetch(*args, **kwargs)

        cache_key = self._get_cache_key(*args, **kwargs)

        # Try to get from cache
        cached_value = await self._cache.get(cache_key)
        if cached_value is not None:
            self.logger.debug(f"Cache hit for key: {cache_key}")
            return cached_value

        # Fetch fresh data
        self.logger.debug(f"Cache miss for key: {cache_key}")
        result = await self.fetch(*args, **kwargs)

        # Store in cache
        await self._cache.set(cache_key, result, ttl_seconds=self.cache_ttl_seconds)

        return result

    async def invalidate_cache(self, *args, **kwargs) -> None:
        """Invalidate cached data for specific arguments."""
        if self._cache is not None:
            cache_key = self._get_cache_key(*args, **kwargs)
            await self._cache.delete(cache_key)
            self.logger.debug(f"Cache invalidated for key: {cache_key}")

    async def clear_all_cache(self) -> None:
        """Clear all cached data for this source."""
        if self._cache is not None:
            await self._cache.clear_prefix(self.cache_prefix)
            self.logger.info(f"All cache cleared for prefix: {self.cache_prefix}")
