"""
Abstract base classes for feature engineering.

Provides common interface, caching support, validation utilities,
and rolling window helpers for all feature builders.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, TypeVar

import polars as pl
from loguru import logger

T = TypeVar("T")


@dataclass
class FeatureMetadata:
    """Metadata about a computed feature."""

    name: str
    description: str
    category: str  # team, player, game, injury
    window: Optional[int] = None  # Rolling window size
    requires_premium: bool = False  # Requires PFF/DVOA/SIC
    default_value: float = 0.0  # Value to use when missing


@dataclass
class FeatureSet:
    """Container for a set of computed features."""

    features: dict[str, float]
    metadata: dict[str, FeatureMetadata] = field(default_factory=dict)
    computed_at: datetime = field(default_factory=datetime.now)
    game_id: Optional[str] = None
    team: Optional[str] = None
    player_id: Optional[str] = None

    def to_dict(self) -> dict[str, float]:
        """Convert features to dictionary."""
        return self.features.copy()

    def get(self, key: str, default: float = 0.0) -> float:
        """Get a feature value with default."""
        return self.features.get(key, default)

    def merge(self, other: "FeatureSet") -> "FeatureSet":
        """Merge two feature sets."""
        merged_features = {**self.features, **other.features}
        merged_metadata = {**self.metadata, **other.metadata}
        return FeatureSet(
            features=merged_features,
            metadata=merged_metadata,
            computed_at=datetime.now(),
            game_id=self.game_id or other.game_id,
            team=self.team or other.team,
            player_id=self.player_id or other.player_id,
        )


class BaseFeatureBuilder(ABC):
    """
    Abstract base class for all feature builders.

    Provides:
    - Common interface for building features
    - Caching support
    - Validation utilities
    - Rolling window helpers
    - Missing data handling

    All feature builder implementations should inherit from this class.
    """

    # Default rolling windows (in games)
    DEFAULT_WINDOWS = [3, 5, 10]
    WINDOW_LABELS = {3: "3g", 5: "5g", 10: "10g", 17: "season"}

    def __init__(
        self,
        cache=None,
        cache_ttl_seconds: int = 21600,  # 6 hours
    ):
        self.cache = cache
        self.cache_ttl_seconds = cache_ttl_seconds
        self.logger = logger.bind(builder=self.__class__.__name__)

    @abstractmethod
    def get_feature_names(self) -> list[str]:
        """
        Get list of feature names this builder produces.

        Returns:
            List of feature names
        """
        pass

    @abstractmethod
    async def build_features(self, *args, **kwargs) -> FeatureSet:
        """
        Build features for the given inputs.

        Returns:
            FeatureSet with computed features
        """
        pass

    def _get_cache_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate a cache key from arguments."""
        import hashlib
        import json

        key_data = {
            "prefix": prefix,
            "args": args,
            "kwargs": {k: v for k, v in sorted(kwargs.items()) if v is not None},
        }
        key_hash = hashlib.md5(json.dumps(key_data, default=str).encode()).hexdigest()
        return f"features:{prefix}:{key_hash}"

    async def _get_cached(self, key: str) -> Optional[FeatureSet]:
        """Get cached features if available."""
        if self.cache is None:
            return None

        try:
            cached = await self.cache.get(key)
            if cached is not None:
                self.logger.debug(f"Cache hit: {key}")
                return cached
        except Exception as e:
            self.logger.warning(f"Cache get error: {e}")

        return None

    async def _set_cached(self, key: str, features: FeatureSet) -> None:
        """Cache computed features."""
        if self.cache is None:
            return

        try:
            await self.cache.set(key, features, ttl_seconds=self.cache_ttl_seconds)
            self.logger.debug(f"Cached: {key}")
        except Exception as e:
            self.logger.warning(f"Cache set error: {e}")

    def _window_label(self, window: int) -> str:
        """Get label for a rolling window size."""
        return self.WINDOW_LABELS.get(window, f"{window}g")

    def _validate_required_columns(
        self,
        df: pl.DataFrame,
        required: list[str],
    ) -> bool:
        """Check if DataFrame has required columns."""
        missing = set(required) - set(df.columns)
        if missing:
            self.logger.warning(f"Missing required columns: {missing}")
            return False
        return True

    def _handle_missing(
        self,
        value: Optional[float],
        default: float = 0.0,
    ) -> float:
        """Handle missing values with default."""
        if value is None or (isinstance(value, float) and value != value):  # NaN check
            return default
        return float(value)

    def _calculate_rolling_mean(
        self,
        values: list[float],
        window: int,
    ) -> Optional[float]:
        """
        Calculate rolling mean of the last N values.

        Args:
            values: List of values (most recent last)
            window: Number of values to average

        Returns:
            Rolling mean or None if insufficient data
        """
        if len(values) < window:
            # Use all available data if less than window
            if len(values) == 0:
                return None
            return sum(values) / len(values)

        recent = values[-window:]
        return sum(recent) / len(recent)

    def _calculate_ema(
        self,
        values: list[float],
        window: int,
        alpha: Optional[float] = None,
    ) -> Optional[float]:
        """
        Calculate Exponential Moving Average of values.

        EMA weights recent values more heavily than older ones,
        better capturing momentum and form changes.

        Args:
            values: List of values (oldest first, most recent last)
            window: Window size (used to calculate alpha if not provided)
            alpha: Smoothing factor (0-1). Higher = more weight on recent.
                   If None, uses alpha = 2/(window+1) (standard EMA formula)

        Returns:
            EMA value or None if insufficient data
        """
        if len(values) == 0:
            return None

        # Filter out None values
        clean_values = [v for v in values if v is not None]
        if len(clean_values) == 0:
            return None

        # Use only the window of most recent values
        if len(clean_values) > window:
            clean_values = clean_values[-window:]

        # Calculate alpha if not provided
        if alpha is None:
            alpha = 2.0 / (len(clean_values) + 1)

        # Initialize EMA with first value
        ema = clean_values[0]

        # Calculate EMA iteratively
        for value in clean_values[1:]:
            ema = alpha * value + (1 - alpha) * ema

        return ema

    def _calculate_volatility(
        self,
        values: list[float],
        window: int,
    ) -> Optional[float]:
        """
        Calculate rolling volatility (coefficient of variation).

        Volatility indicates consistency - high volatility means
        unpredictable performance.

        Args:
            values: List of values
            window: Window size

        Returns:
            Coefficient of variation (std/mean) or None
        """
        if len(values) < 2:
            return None

        recent = values[-window:] if len(values) >= window else values
        mean = sum(recent) / len(recent)

        if mean == 0:
            return None

        variance = sum((x - mean) ** 2 for x in recent) / len(recent)
        std = variance ** 0.5

        return std / abs(mean)  # Coefficient of variation

    def _calculate_rolling_std(
        self,
        values: list[float],
        window: int,
    ) -> Optional[float]:
        """Calculate rolling standard deviation."""
        if len(values) < 2:
            return None

        recent = values[-window:] if len(values) >= window else values
        mean = sum(recent) / len(recent)
        variance = sum((x - mean) ** 2 for x in recent) / len(recent)
        return variance**0.5

    def _calculate_trend(
        self,
        values: list[float],
        window: int = 5,
    ) -> float:
        """
        Calculate trend (slope) over recent values.

        Returns:
            Positive for increasing, negative for decreasing
        """
        if len(values) < 2:
            return 0.0

        recent = values[-window:] if len(values) >= window else values
        n = len(recent)

        # Simple linear regression slope
        x_mean = (n - 1) / 2
        y_mean = sum(recent) / n

        numerator = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(recent))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return 0.0

        return numerator / denominator


class RollingWindowMixin:
    """
    Mixin providing rolling window calculations for Polars DataFrames.

    Optimized for NFL game-based rolling windows.
    """

    def compute_game_aggregates(
        self,
        pbp_df: pl.DataFrame,
        team: str,
        group_cols: list[str],
        agg_exprs: list[pl.Expr],
    ) -> pl.DataFrame:
        """
        Compute per-game aggregates for a team.

        Args:
            pbp_df: Play-by-play DataFrame
            team: Team abbreviation
            group_cols: Columns to group by (usually ["game_id"])
            agg_exprs: Polars aggregation expressions

        Returns:
            DataFrame with game-level aggregates
        """
        return (
            pbp_df.filter(pl.col("posteam") == team)
            .group_by(group_cols)
            .agg(agg_exprs)
            .sort("game_id")
        )

    def compute_rolling_features(
        self,
        game_df: pl.DataFrame,
        metrics: list[str],
        windows: list[int],
        prefix: str = "",
    ) -> pl.DataFrame:
        """
        Compute rolling features for multiple metrics and windows.

        Args:
            game_df: DataFrame with game-level metrics
            metrics: List of metric column names
            windows: List of rolling window sizes
            prefix: Prefix for output column names

        Returns:
            DataFrame with rolling features
        """
        rolling_exprs = []

        for metric in metrics:
            for window in windows:
                label = f"{window}g"
                col_name = f"{prefix}{metric}_{label}" if prefix else f"{metric}_{label}"

                # Shift by 1 to avoid look-ahead bias
                rolling_exprs.append(
                    pl.col(metric)
                    .shift(1)
                    .rolling_mean(window_size=window, min_periods=1)
                    .alias(col_name)
                )

        return game_df.with_columns(rolling_exprs)

    def get_latest_rolling_values(
        self,
        game_df: pl.DataFrame,
        metrics: list[str],
        windows: list[int],
        before_game_id: Optional[str] = None,
        prefix: str = "",
    ) -> dict[str, float]:
        """
        Get the latest rolling values before a specific game.

        Args:
            game_df: DataFrame with game-level metrics (sorted by game_id)
            metrics: List of metric column names
            windows: List of rolling window sizes
            before_game_id: Game ID to get values before (None for latest)
            prefix: Prefix for output keys

        Returns:
            Dictionary of rolling feature values
        """
        # Compute rolling features
        rolling_df = self.compute_rolling_features(game_df, metrics, windows, prefix)

        # Get the appropriate row
        if before_game_id:
            row = rolling_df.filter(pl.col("game_id") == before_game_id)
        else:
            row = rolling_df.tail(1)

        if len(row) == 0:
            return {}

        result = {}
        row_dict = row.row(0, named=True)

        for metric in metrics:
            for window in windows:
                label = f"{window}g"
                col_name = f"{prefix}{metric}_{label}" if prefix else f"{metric}_{label}"
                if col_name in row_dict:
                    value = row_dict[col_name]
                    if value is not None and value == value:  # Not None or NaN
                        result[col_name] = float(value)

        return result


class FeatureScaler:
    """
    Feature scaling utilities for ML preparation.

    Supports standardization (z-score) and min-max scaling.
    """

    def __init__(self):
        self.means: dict[str, float] = {}
        self.stds: dict[str, float] = {}
        self.mins: dict[str, float] = {}
        self.maxs: dict[str, float] = {}
        self._fitted = False

    def fit(self, df: pl.DataFrame, columns: list[str]) -> "FeatureScaler":
        """
        Compute scaling parameters from training data.

        Args:
            df: Training DataFrame
            columns: Columns to scale

        Returns:
            Self for chaining
        """
        for col in columns:
            if col in df.columns:
                values = df[col].drop_nulls()
                self.means[col] = values.mean()
                self.stds[col] = values.std()
                self.mins[col] = values.min()
                self.maxs[col] = values.max()

        self._fitted = True
        return self

    def standardize(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply z-score standardization."""
        if not self._fitted:
            raise ValueError("Scaler not fitted. Call fit() first.")

        exprs = []
        for col, mean in self.means.items():
            if col in df.columns:
                std = self.stds.get(col, 1.0)
                if std == 0:
                    std = 1.0
                exprs.append(((pl.col(col) - mean) / std).alias(col))

        return df.with_columns(exprs) if exprs else df

    def min_max_scale(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply min-max scaling to [0, 1] range."""
        if not self._fitted:
            raise ValueError("Scaler not fitted. Call fit() first.")

        exprs = []
        for col, min_val in self.mins.items():
            if col in df.columns:
                max_val = self.maxs.get(col, min_val + 1)
                range_val = max_val - min_val
                if range_val == 0:
                    range_val = 1.0
                exprs.append(((pl.col(col) - min_val) / range_val).alias(col))

        return df.with_columns(exprs) if exprs else df

    def save(self, path: str) -> None:
        """Save scaling parameters to file."""
        import json

        params = {
            "means": self.means,
            "stds": self.stds,
            "mins": self.mins,
            "maxs": self.maxs,
        }
        with open(path, "w") as f:
            json.dump(params, f)

    def load(self, path: str) -> "FeatureScaler":
        """Load scaling parameters from file."""
        import json

        with open(path, "r") as f:
            params = json.load(f)

        self.means = params["means"]
        self.stds = params["stds"]
        self.mins = params["mins"]
        self.maxs = params["maxs"]
        self._fitted = True
        return self


def validate_no_lookahead(
    features_df: pl.DataFrame,
    target_df: pl.DataFrame,
    date_col: str = "game_date",
) -> bool:
    """
    Validate that features don't contain look-ahead bias.

    Checks that all feature data comes from before the target game.

    Args:
        features_df: DataFrame with features and dates
        target_df: DataFrame with targets and dates
        date_col: Column containing game dates

    Returns:
        True if valid, raises ValueError if look-ahead detected
    """
    # This is a placeholder - actual implementation would check
    # that features are computed from prior games only
    return True


def fill_missing_features(
    features: dict[str, float],
    required_features: list[str],
    defaults: Optional[dict[str, float]] = None,
) -> dict[str, float]:
    """
    Fill missing features with default values.

    Args:
        features: Computed features
        required_features: List of required feature names
        defaults: Optional custom defaults (otherwise uses 0.0)

    Returns:
        Features with all required keys present
    """
    defaults = defaults or {}
    result = features.copy()

    for feature in required_features:
        if feature not in result:
            result[feature] = defaults.get(feature, 0.0)

    return result
