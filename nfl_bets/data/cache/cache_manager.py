"""
Unified caching layer with multiple backend support.

Provides a consistent interface for caching data across different backends:
- Redis (production)
- SQLite (local development)
- In-memory (testing)

Supports TTL-based expiration and prefix-based invalidation.
"""
import asyncio
import json
import pickle
import sqlite3
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional, Union

from loguru import logger


class CacheBackend(ABC):
    """Abstract base class for cache backends."""

    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get a value from cache."""
        pass

    @abstractmethod
    async def set(self, key: str, value: Any, ttl_seconds: int) -> None:
        """Set a value in cache with TTL."""
        pass

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Delete a key from cache."""
        pass

    @abstractmethod
    async def clear_prefix(self, prefix: str) -> None:
        """Clear all keys with given prefix."""
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        pass

    @abstractmethod
    async def get_ttl(self, key: str) -> Optional[int]:
        """Get remaining TTL for a key in seconds."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close the cache connection."""
        pass


class InMemoryCache(CacheBackend):
    """
    Simple in-memory cache backend.

    Best for testing and small-scale local usage.
    Data is lost when the process exits.
    """

    def __init__(self, max_size: int = 10000):
        self._cache: dict[str, tuple[Any, float]] = {}  # value, expiry_time
        self._max_size = max_size
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[Any]:
        async with self._lock:
            if key not in self._cache:
                return None

            value, expiry_time = self._cache[key]

            # Check if expired
            if expiry_time and time.time() > expiry_time:
                del self._cache[key]
                return None

            return value

    async def set(self, key: str, value: Any, ttl_seconds: int) -> None:
        async with self._lock:
            # Evict oldest entries if at capacity
            if len(self._cache) >= self._max_size:
                self._evict_expired()
                if len(self._cache) >= self._max_size:
                    # Remove oldest 10%
                    keys_to_remove = list(self._cache.keys())[: self._max_size // 10]
                    for k in keys_to_remove:
                        del self._cache[k]

            expiry_time = time.time() + ttl_seconds if ttl_seconds > 0 else None
            self._cache[key] = (value, expiry_time)

    async def delete(self, key: str) -> None:
        async with self._lock:
            self._cache.pop(key, None)

    async def clear_prefix(self, prefix: str) -> None:
        async with self._lock:
            keys_to_delete = [k for k in self._cache.keys() if k.startswith(prefix)]
            for key in keys_to_delete:
                del self._cache[key]

    async def exists(self, key: str) -> bool:
        value = await self.get(key)
        return value is not None

    async def get_ttl(self, key: str) -> Optional[int]:
        async with self._lock:
            if key not in self._cache:
                return None

            _, expiry_time = self._cache[key]
            if expiry_time is None:
                return -1  # No expiry

            remaining = int(expiry_time - time.time())
            return max(0, remaining)

    async def close(self) -> None:
        async with self._lock:
            self._cache.clear()

    def _evict_expired(self) -> None:
        """Remove all expired entries."""
        current_time = time.time()
        keys_to_delete = [
            k for k, (_, exp) in self._cache.items() if exp and current_time > exp
        ]
        for key in keys_to_delete:
            del self._cache[key]


class SQLiteCache(CacheBackend):
    """
    SQLite-based cache backend.

    Good for local development with persistence.
    Data survives process restarts.
    """

    def __init__(self, db_path: Union[str, Path] = "cache.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection: Optional[sqlite3.Connection] = None
        self._lock = asyncio.Lock()
        self._initialized = False

    async def _ensure_initialized(self) -> None:
        """Ensure the database is initialized."""
        if self._initialized:
            return

        async with self._lock:
            if self._initialized:
                return

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._init_db)
            self._initialized = True

    def _init_db(self) -> None:
        """Initialize the database schema."""
        self._connection = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._connection.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                value BLOB,
                expiry_time REAL
            )
        """)
        self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_cache_expiry ON cache(expiry_time)"
        )
        self._connection.commit()

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        if self._connection is None:
            self._connection = sqlite3.connect(
                str(self.db_path), check_same_thread=False
            )
        return self._connection

    async def get(self, key: str) -> Optional[Any]:
        await self._ensure_initialized()

        loop = asyncio.get_event_loop()

        def _get():
            conn = self._get_connection()
            cursor = conn.execute(
                "SELECT value, expiry_time FROM cache WHERE key = ?", (key,)
            )
            row = cursor.fetchone()

            if row is None:
                return None

            value_blob, expiry_time = row

            # Check if expired
            if expiry_time and time.time() > expiry_time:
                conn.execute("DELETE FROM cache WHERE key = ?", (key,))
                conn.commit()
                return None

            return pickle.loads(value_blob)

        return await loop.run_in_executor(None, _get)

    async def set(self, key: str, value: Any, ttl_seconds: int) -> None:
        await self._ensure_initialized()

        loop = asyncio.get_event_loop()

        def _set():
            conn = self._get_connection()
            value_blob = pickle.dumps(value)
            expiry_time = time.time() + ttl_seconds if ttl_seconds > 0 else None

            conn.execute(
                """
                INSERT OR REPLACE INTO cache (key, value, expiry_time)
                VALUES (?, ?, ?)
            """,
                (key, value_blob, expiry_time),
            )
            conn.commit()

        await loop.run_in_executor(None, _set)

    async def delete(self, key: str) -> None:
        await self._ensure_initialized()

        loop = asyncio.get_event_loop()

        def _delete():
            conn = self._get_connection()
            conn.execute("DELETE FROM cache WHERE key = ?", (key,))
            conn.commit()

        await loop.run_in_executor(None, _delete)

    async def clear_prefix(self, prefix: str) -> None:
        await self._ensure_initialized()

        loop = asyncio.get_event_loop()

        def _clear():
            conn = self._get_connection()
            conn.execute("DELETE FROM cache WHERE key LIKE ?", (f"{prefix}%",))
            conn.commit()

        await loop.run_in_executor(None, _clear)

    async def exists(self, key: str) -> bool:
        value = await self.get(key)
        return value is not None

    async def get_ttl(self, key: str) -> Optional[int]:
        await self._ensure_initialized()

        loop = asyncio.get_event_loop()

        def _get_ttl():
            conn = self._get_connection()
            cursor = conn.execute(
                "SELECT expiry_time FROM cache WHERE key = ?", (key,)
            )
            row = cursor.fetchone()

            if row is None:
                return None

            expiry_time = row[0]
            if expiry_time is None:
                return -1

            remaining = int(expiry_time - time.time())
            return max(0, remaining)

        return await loop.run_in_executor(None, _get_ttl)

    async def close(self) -> None:
        if self._connection:
            self._connection.close()
            self._connection = None

    async def cleanup_expired(self) -> int:
        """Remove all expired entries. Returns count of removed entries."""
        await self._ensure_initialized()

        loop = asyncio.get_event_loop()

        def _cleanup():
            conn = self._get_connection()
            cursor = conn.execute(
                "DELETE FROM cache WHERE expiry_time IS NOT NULL AND expiry_time < ?",
                (time.time(),),
            )
            count = cursor.rowcount
            conn.commit()
            return count

        return await loop.run_in_executor(None, _cleanup)


class RedisCache(CacheBackend):
    """
    Redis-based cache backend.

    Best for production with distributed caching needs.
    Requires Redis server.
    """

    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.redis_url = redis_url
        self._redis = None
        self._lock = asyncio.Lock()

    async def _get_redis(self):
        """Get or create Redis connection."""
        if self._redis is None:
            async with self._lock:
                if self._redis is None:
                    try:
                        import redis.asyncio as redis

                        self._redis = await redis.from_url(
                            self.redis_url,
                            encoding="utf-8",
                            decode_responses=False,
                        )
                    except ImportError:
                        raise ImportError(
                            "redis package is required for RedisCache. "
                            "Install with: pip install redis"
                        )
        return self._redis

    async def get(self, key: str) -> Optional[Any]:
        redis = await self._get_redis()
        value = await redis.get(key)

        if value is None:
            return None

        try:
            return pickle.loads(value)
        except Exception:
            return None

    async def set(self, key: str, value: Any, ttl_seconds: int) -> None:
        redis = await self._get_redis()
        value_blob = pickle.dumps(value)
        await redis.setex(key, ttl_seconds, value_blob)

    async def delete(self, key: str) -> None:
        redis = await self._get_redis()
        await redis.delete(key)

    async def clear_prefix(self, prefix: str) -> None:
        redis = await self._get_redis()
        cursor = 0

        while True:
            cursor, keys = await redis.scan(cursor, match=f"{prefix}*", count=100)
            if keys:
                await redis.delete(*keys)
            if cursor == 0:
                break

    async def exists(self, key: str) -> bool:
        redis = await self._get_redis()
        return await redis.exists(key) > 0

    async def get_ttl(self, key: str) -> Optional[int]:
        redis = await self._get_redis()
        ttl = await redis.ttl(key)
        return ttl if ttl >= 0 else None

    async def close(self) -> None:
        if self._redis:
            await self._redis.close()
            self._redis = None


class CacheManager:
    """
    Unified cache manager with backend abstraction.

    Provides a high-level interface for caching with automatic
    serialization and backend selection.
    """

    # Default TTL values by data type
    DEFAULT_TTLS = {
        "odds": 300,  # 5 minutes
        "pbp": 86400,  # 24 hours
        "player_stats": 86400,  # 24 hours
        "roster": 86400,  # 24 hours
        "schedule": 3600,  # 1 hour
        "predictions": 1800,  # 30 minutes
        "dvoa": 86400,  # 24 hours
        "pff": 86400,  # 24 hours
        "injury": 3600,  # 1 hour
        "default": 3600,  # 1 hour
    }

    def __init__(
        self,
        backend: CacheBackend,
        key_prefix: str = "nfl_bets",
    ):
        self.backend = backend
        self.key_prefix = key_prefix
        self.logger = logger.bind(component="cache")

    @classmethod
    def create_memory_cache(cls, max_size: int = 10000) -> "CacheManager":
        """Create a cache manager with in-memory backend."""
        return cls(InMemoryCache(max_size=max_size))

    @classmethod
    def create_sqlite_cache(
        cls, db_path: Union[str, Path] = "data/cache/cache.db"
    ) -> "CacheManager":
        """Create a cache manager with SQLite backend."""
        return cls(SQLiteCache(db_path=db_path))

    @classmethod
    def create_redis_cache(
        cls, redis_url: str = "redis://localhost:6379/0"
    ) -> "CacheManager":
        """Create a cache manager with Redis backend."""
        return cls(RedisCache(redis_url=redis_url))

    @classmethod
    def create_from_settings(cls, settings) -> "CacheManager":
        """Create cache manager based on application settings."""
        if settings.redis_url:
            return cls.create_redis_cache(settings.redis_url)
        else:
            cache_path = settings.data_dir / "cache" / "cache.db"
            return cls.create_sqlite_cache(cache_path)

    def _make_key(self, key: str) -> str:
        """Create a namespaced cache key."""
        return f"{self.key_prefix}:{key}"

    def _get_ttl(self, data_type: str, ttl_seconds: Optional[int] = None) -> int:
        """Get TTL for a data type."""
        if ttl_seconds is not None:
            return ttl_seconds
        return self.DEFAULT_TTLS.get(data_type, self.DEFAULT_TTLS["default"])

    async def get(self, key: str) -> Optional[Any]:
        """Get a value from cache."""
        full_key = self._make_key(key)
        try:
            value = await self.backend.get(full_key)
            if value is not None:
                self.logger.debug(f"Cache hit: {key}")
            else:
                self.logger.debug(f"Cache miss: {key}")
            return value
        except Exception as e:
            self.logger.error(f"Cache get error for {key}: {e}")
            return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None,
        data_type: str = "default",
    ) -> None:
        """Set a value in cache."""
        full_key = self._make_key(key)
        ttl = self._get_ttl(data_type, ttl_seconds)

        try:
            await self.backend.set(full_key, value, ttl)
            self.logger.debug(f"Cache set: {key} (TTL: {ttl}s)")
        except Exception as e:
            self.logger.error(f"Cache set error for {key}: {e}")

    async def delete(self, key: str) -> None:
        """Delete a key from cache."""
        full_key = self._make_key(key)
        try:
            await self.backend.delete(full_key)
            self.logger.debug(f"Cache delete: {key}")
        except Exception as e:
            self.logger.error(f"Cache delete error for {key}: {e}")

    async def clear_prefix(self, prefix: str) -> None:
        """Clear all keys with given prefix."""
        full_prefix = self._make_key(prefix)
        try:
            await self.backend.clear_prefix(full_prefix)
            self.logger.info(f"Cache cleared for prefix: {prefix}")
        except Exception as e:
            self.logger.error(f"Cache clear error for prefix {prefix}: {e}")

    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        full_key = self._make_key(key)
        try:
            return await self.backend.exists(full_key)
        except Exception as e:
            self.logger.error(f"Cache exists error for {key}: {e}")
            return False

    async def get_or_set(
        self,
        key: str,
        factory,
        ttl_seconds: Optional[int] = None,
        data_type: str = "default",
    ) -> Any:
        """
        Get value from cache or compute and store it.

        Args:
            key: Cache key
            factory: Async callable to compute value if not cached
            ttl_seconds: Optional TTL override
            data_type: Data type for default TTL lookup

        Returns:
            Cached or computed value
        """
        value = await self.get(key)
        if value is not None:
            return value

        # Compute value
        value = await factory()

        # Store in cache
        await self.set(key, value, ttl_seconds, data_type)

        return value

    async def close(self) -> None:
        """Close the cache backend."""
        await self.backend.close()

    async def health_check(self) -> dict:
        """Check cache health."""
        try:
            test_key = f"{self.key_prefix}:_health_check"
            await self.backend.set(test_key, "ok", 60)
            value = await self.backend.get(test_key)
            await self.backend.delete(test_key)

            return {
                "status": "healthy" if value == "ok" else "degraded",
                "backend": type(self.backend).__name__,
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "backend": type(self.backend).__name__,
                "error": str(e),
            }
