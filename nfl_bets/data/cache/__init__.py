"""
Caching layer for NFL betting system.

Provides multiple cache backends:
- InMemoryCache: Fast, ephemeral cache for testing
- SQLiteCache: Persistent local cache
- RedisCache: Distributed cache for production
"""
from .cache_manager import (
    CacheBackend,
    CacheManager,
    InMemoryCache,
    SQLiteCache,
    RedisCache,
)

__all__ = [
    "CacheBackend",
    "CacheManager",
    "InMemoryCache",
    "SQLiteCache",
    "RedisCache",
]
