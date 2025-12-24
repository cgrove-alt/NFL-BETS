"""
Feature engineering module for NFL betting predictions.

This module provides feature builders for:
- Team-level features (EPA, efficiency, situational)
- Player-level features (volume, efficiency, matchups)
- Game context features (rest, travel, weather, Vegas)
- Injury impact features (SIC Score health integration)

Example:
    >>> from nfl_bets.features import FeaturePipeline, create_feature_pipeline
    >>> pipeline = await create_feature_pipeline(data_pipeline)
    >>> features = await pipeline.build_spread_features(
    ...     game_id="2023_05_KC_BUF",
    ...     home_team="BUF",
    ...     away_team="KC",
    ...     season=2023,
    ...     week=5
    ... )
"""

from .base import (
    BaseFeatureBuilder,
    FeatureSet,
    FeatureScaler,
    RollingWindowMixin,
    fill_missing_features,
    validate_feature_names,
)
from .team_features import TeamFeatureBuilder
from .player_features import PlayerFeatureBuilder
from .game_features import GameFeatureBuilder
from .injury_features import (
    InjuryFeatureBuilder,
    calculate_injury_adjusted_projection,
    estimate_injury_variance,
)
from .feature_pipeline import (
    FeaturePipeline,
    SpreadPredictionFeatures,
    PropPredictionFeatures,
    create_feature_pipeline,
)

__all__ = [
    # Base classes
    "BaseFeatureBuilder",
    "FeatureSet",
    "FeatureScaler",
    "RollingWindowMixin",
    "fill_missing_features",
    "validate_feature_names",
    # Feature builders
    "TeamFeatureBuilder",
    "PlayerFeatureBuilder",
    "GameFeatureBuilder",
    "InjuryFeatureBuilder",
    # Injury utilities
    "calculate_injury_adjusted_projection",
    "estimate_injury_variance",
    # Pipeline
    "FeaturePipeline",
    "SpreadPredictionFeatures",
    "PropPredictionFeatures",
    "create_feature_pipeline",
]
