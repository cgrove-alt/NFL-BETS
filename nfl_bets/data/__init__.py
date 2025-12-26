"""
Data layer for NFL betting system.

Provides unified access to all data sources:
- nflverse (play-by-play, stats, schedules)
- The Odds API (live betting odds)
- PFF (player grades)
- FTN/DVOA (team efficiency)
- SIC Score (injury data)
- Odds history database (line tracking, CLV)
"""
from .pipeline import DataPipeline, GameData, PipelineHealth, create_pipeline
from .odds_history import (
    OddsHistoryDB,
    OddsSnapshot,
    CLVResult,
    BetType,
    check_clv,
)

__all__ = [
    # Pipeline
    "DataPipeline",
    "GameData",
    "PipelineHealth",
    "create_pipeline",
    # Odds History
    "OddsHistoryDB",
    "OddsSnapshot",
    "CLVResult",
    "BetType",
    "check_clv",
]
