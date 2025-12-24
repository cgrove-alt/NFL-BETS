"""
Data source clients for NFL betting system.

Available sources:
- NFLVerseClient: nflreadpy integration for play-by-play and stats
- OddsAPIClient: The Odds API for real-time betting odds
- PFFClient: Pro Football Focus player grades
- FTNDVOAClient: Football Outsiders DVOA metrics
- SICScoreClient: Sports Injury Central injury data
"""
from .base import (
    BaseDataSource,
    CachedDataSource,
    DataSourceError,
    DataSourceHealth,
    DataSourceStatus,
    RateLimitError,
    AuthenticationError,
    DataNotAvailableError,
    RetryConfig,
    CircuitBreakerConfig,
)
from .nflverse import NFLVerseClient
from .odds_api import OddsAPIClient, OddsAPIClientFactory
from .pff_client import PFFClient, create_pff_client_from_settings
from .ftn_dvoa import FTNDVOAClient, create_ftn_client_from_settings
from .sic_score import SICScoreClient, create_sic_client_from_settings

__all__ = [
    # Base classes
    "BaseDataSource",
    "CachedDataSource",
    "DataSourceError",
    "DataSourceHealth",
    "DataSourceStatus",
    "RateLimitError",
    "AuthenticationError",
    "DataNotAvailableError",
    "RetryConfig",
    "CircuitBreakerConfig",
    # Clients
    "NFLVerseClient",
    "OddsAPIClient",
    "OddsAPIClientFactory",
    "PFFClient",
    "FTNDVOAClient",
    "SICScoreClient",
    # Factory functions
    "create_pff_client_from_settings",
    "create_ftn_client_from_settings",
    "create_sic_client_from_settings",
]
