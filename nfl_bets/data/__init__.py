"""
Data layer for NFL betting system.

Provides unified access to all data sources:
- nflverse (play-by-play, stats, schedules)
- The Odds API (live betting odds)
- PFF (player grades)
- FTN/DVOA (team efficiency)
- SIC Score (injury data)
"""
from .pipeline import DataPipeline, GameData, PipelineHealth, create_pipeline

__all__ = [
    "DataPipeline",
    "GameData",
    "PipelineHealth",
    "create_pipeline",
]
