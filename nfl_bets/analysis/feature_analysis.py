"""
Feature importance and stability analysis.

Provides:
- Feature importance with confidence intervals
- Stability across folds/seasons
- Feature correlation analysis
- Importance trend tracking
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from loguru import logger


@dataclass
class FeatureImportance:
    """Importance metrics for a single feature."""

    name: str
    importance: float
    rank: int

    # Stability metrics
    std: float  # Standard deviation across folds
    cv: float  # Coefficient of variation
    ci_lower: float
    ci_upper: float

    # Trend
    trend: float  # Change in importance over time (+ = increasing)

    @property
    def is_stable(self) -> bool:
        """Is feature importance stable across folds?"""
        return self.cv < 0.5  # Less than 50% variation

    @property
    def is_significant(self) -> bool:
        """Is feature significantly important (CI doesn't include 0)?"""
        return self.ci_lower > 0


@dataclass
class FeatureStability:
    """Stability analysis across folds."""

    feature_name: str
    fold_importances: list[float]

    @property
    def mean(self) -> float:
        return float(np.mean(self.fold_importances))

    @property
    def std(self) -> float:
        return float(np.std(self.fold_importances))

    @property
    def cv(self) -> float:
        """Coefficient of variation."""
        if self.mean == 0:
            return float('inf')
        return self.std / abs(self.mean)

    @property
    def is_stable(self) -> bool:
        return self.cv < 0.5

    @property
    def rank_variance(self) -> float:
        """Variance in rankings across folds."""
        if len(self.fold_importances) < 2:
            return 0.0
        # Convert importances to ranks
        ranks = []
        for imp in self.fold_importances:
            ranks.append(imp)  # Simplified - actual rank needs all features
        return float(np.var(ranks))


@dataclass
class FeatureCorrelation:
    """Correlation between features."""

    feature_a: str
    feature_b: str
    correlation: float

    @property
    def is_redundant(self) -> bool:
        """Are features potentially redundant (highly correlated)?"""
        return abs(self.correlation) > 0.8


@dataclass
class FeatureImportanceReport:
    """Complete feature importance analysis."""

    model_name: str
    n_features: int
    n_folds: int

    # Ranked importance
    features: list[FeatureImportance]

    # Stability analysis
    stability: dict[str, FeatureStability]

    # Correlations
    top_correlations: list[FeatureCorrelation]

    # Summary metrics
    concentration: float  # How concentrated is importance (Gini)
    n_stable_features: int
    n_redundant_pairs: int

    def get_top_features(self, n: int = 10) -> list[FeatureImportance]:
        """Get top N most important features."""
        return self.features[:n]

    def get_unstable_features(self) -> list[FeatureImportance]:
        """Get features with unstable importance."""
        return [f for f in self.features if not f.is_stable]

    def summary(self) -> str:
        """Generate summary string."""
        if self.n_features == 0:
            return f"Feature Importance Report: {self.model_name}\n" + "=" * 60 + "\nNo features available for analysis."

        stable_pct = 100 * self.n_stable_features / self.n_features if self.n_features > 0 else 0
        lines = [
            f"Feature Importance Report: {self.model_name}",
            "=" * 60,
            f"Total Features: {self.n_features}",
            f"Folds Analyzed: {self.n_folds}",
            f"Stable Features: {self.n_stable_features} ({stable_pct:.0f}%)",
            f"Importance Concentration (Gini): {self.concentration:.3f}",
            "",
            "TOP 10 FEATURES:",
            "| Rank | Feature | Importance | Std | Stable | Trend |",
            "|------|---------|------------|-----|--------|-------|",
        ]

        for f in self.features[:10]:
            stable = "✓" if f.is_stable else "✗"
            trend_arrow = "↑" if f.trend > 0.01 else ("↓" if f.trend < -0.01 else "→")
            lines.append(
                f"| {f.rank} | {f.name[:25]} | {f.importance:.4f} | "
                f"{f.std:.4f} | {stable} | {trend_arrow} |"
            )

        if self.top_correlations:
            lines.append("")
            lines.append("HIGHLY CORRELATED FEATURE PAIRS:")
            for corr in self.top_correlations[:5]:
                lines.append(f"  {corr.feature_a} ↔ {corr.feature_b}: {corr.correlation:.3f}")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "model_name": self.model_name,
            "n_features": self.n_features,
            "n_folds": self.n_folds,
            "concentration": self.concentration,
            "features": [
                {
                    "name": f.name,
                    "importance": f.importance,
                    "rank": f.rank,
                    "std": f.std,
                    "is_stable": f.is_stable,
                }
                for f in self.features
            ],
        }


class FeatureAnalyzer:
    """
    Comprehensive feature importance analysis.

    Analyzes:
    - Feature importance from trained models
    - Stability across validation folds
    - Feature correlations and redundancy
    - Importance trends over time
    """

    def __init__(self, min_folds: int = 3):
        self.min_folds = min_folds
        self.logger = logger.bind(component="feature_analyzer")

    def analyze(
        self,
        fold_importances: list[dict[str, float]],
        feature_matrix: Optional[np.ndarray] = None,
        feature_names: Optional[list[str]] = None,
        model_name: str = "Model",
    ) -> FeatureImportanceReport:
        """
        Perform comprehensive feature importance analysis.

        Args:
            fold_importances: List of {feature: importance} dicts from each fold
            feature_matrix: Optional X matrix for correlation analysis
            feature_names: Feature names (if matrix provided)
            model_name: Model name for reporting

        Returns:
            FeatureImportanceReport with full analysis
        """
        if len(fold_importances) < self.min_folds:
            self.logger.warning(f"Only {len(fold_importances)} folds, need {self.min_folds}")

        n_folds = len(fold_importances)

        # Get all feature names
        all_features = set()
        for fold in fold_importances:
            all_features.update(fold.keys())

        all_features = sorted(all_features)
        n_features = len(all_features)

        self.logger.info(f"Analyzing {n_features} features across {n_folds} folds")

        # Calculate stability for each feature
        stability = {}
        for feature in all_features:
            importances = [
                fold.get(feature, 0.0) for fold in fold_importances
            ]
            stability[feature] = FeatureStability(
                feature_name=feature,
                fold_importances=importances,
            )

        # Calculate aggregate importance with CI
        features = []
        for i, feature in enumerate(sorted(all_features, key=lambda f: stability[f].mean, reverse=True)):
            stab = stability[feature]

            # Bootstrap CI
            if n_folds >= 3:
                ci = self._bootstrap_ci(stab.fold_importances)
            else:
                ci = (stab.mean - 1.96 * stab.std, stab.mean + 1.96 * stab.std)

            # Trend (simplified - first half vs second half)
            if n_folds >= 4:
                first_half = stab.fold_importances[:n_folds//2]
                second_half = stab.fold_importances[n_folds//2:]
                trend = np.mean(second_half) - np.mean(first_half)
            else:
                trend = 0.0

            features.append(FeatureImportance(
                name=feature,
                importance=stab.mean,
                rank=i + 1,
                std=stab.std,
                cv=stab.cv if stab.cv != float('inf') else 999,
                ci_lower=ci[0],
                ci_upper=ci[1],
                trend=trend,
            ))

        # Correlation analysis
        top_correlations = []
        if feature_matrix is not None and feature_names is not None:
            top_correlations = self._analyze_correlations(
                feature_matrix, feature_names
            )

        # Summary metrics
        importances = [f.importance for f in features]
        concentration = self._gini_coefficient(importances)
        n_stable = sum(1 for f in features if f.is_stable)
        n_redundant = sum(1 for c in top_correlations if c.is_redundant)

        return FeatureImportanceReport(
            model_name=model_name,
            n_features=n_features,
            n_folds=n_folds,
            features=features,
            stability=stability,
            top_correlations=top_correlations,
            concentration=concentration,
            n_stable_features=n_stable,
            n_redundant_pairs=n_redundant,
        )

    def analyze_from_model(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
        n_folds: int = 5,
        model_name: str = "Model",
    ) -> FeatureImportanceReport:
        """
        Analyze feature importance by retraining across folds.

        Args:
            model: Model class (will be instantiated per fold)
            X: Feature matrix
            y: Target values
            feature_names: Feature names
            n_folds: Number of cross-validation folds
            model_name: Model name for reporting

        Returns:
            FeatureImportanceReport
        """
        import polars as pl
        from sklearn.model_selection import KFold

        fold_importances = []
        kf = KFold(n_splits=n_folds, shuffle=False)

        for fold_idx, (train_idx, _) in enumerate(kf.split(X)):
            self.logger.debug(f"Training fold {fold_idx + 1}/{n_folds}")

            X_train = X[train_idx]
            y_train = y[train_idx]

            # Convert to DataFrame to properly set feature names
            X_train_df = pl.DataFrame(X_train, schema=feature_names)

            # Train model
            fold_model = model.__class__()
            fold_model.train(X_train_df, y_train)

            # Get importance
            try:
                importance = fold_model.get_feature_importance()
                # Map to feature names if needed
                if isinstance(importance, dict):
                    if importance:  # Only add if not empty
                        fold_importances.append(importance)
                    else:
                        self.logger.warning(f"Fold {fold_idx}: get_feature_importance returned empty dict")
                else:
                    fold_importances.append({
                        name: imp for name, imp in zip(feature_names, importance)
                    })
            except Exception as e:
                self.logger.warning(f"Could not get importance for fold {fold_idx}: {e}")

        return self.analyze(
            fold_importances=fold_importances,
            feature_matrix=X,
            feature_names=feature_names,
            model_name=model_name,
        )

    def _bootstrap_ci(
        self,
        values: list[float],
        n_bootstrap: int = 1000,
        confidence: float = 0.95,
    ) -> tuple[float, float]:
        """Calculate bootstrap confidence interval."""
        values = np.array(values)
        n = len(values)

        if n == 0:
            return (0.0, 0.0)

        bootstrap_means = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(values, size=n, replace=True)
            bootstrap_means.append(np.mean(sample))

        alpha = 1 - confidence
        ci_lower = np.percentile(bootstrap_means, alpha / 2 * 100)
        ci_upper = np.percentile(bootstrap_means, (1 - alpha / 2) * 100)

        return (float(ci_lower), float(ci_upper))

    def _analyze_correlations(
        self,
        X: np.ndarray,
        feature_names: list[str],
        top_n: int = 20,
    ) -> list[FeatureCorrelation]:
        """Find highly correlated feature pairs."""
        if X.shape[1] != len(feature_names):
            self.logger.warning("Feature names don't match matrix dimensions")
            return []

        # Calculate correlation matrix
        corr_matrix = np.corrcoef(X.T)

        # Find top correlations (excluding diagonal)
        correlations = []
        for i in range(len(feature_names)):
            for j in range(i + 1, len(feature_names)):
                corr = corr_matrix[i, j]
                if not np.isnan(corr):
                    correlations.append(FeatureCorrelation(
                        feature_a=feature_names[i],
                        feature_b=feature_names[j],
                        correlation=float(corr),
                    ))

        # Sort by absolute correlation
        correlations.sort(key=lambda c: abs(c.correlation), reverse=True)

        return correlations[:top_n]

    def _gini_coefficient(self, values: list[float]) -> float:
        """
        Calculate Gini coefficient for importance concentration.

        0 = perfectly equal, 1 = perfectly concentrated
        """
        values = np.array(sorted(values))
        n = len(values)

        if n == 0 or np.sum(values) == 0:
            return 0.0

        cumulative = np.cumsum(values)
        gini = (n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n

        return float(gini)


def permutation_importance(
    model,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    n_repeats: int = 10,
    scoring: str = "mae",
) -> dict[str, float]:
    """
    Calculate permutation importance.

    Measures how much performance degrades when a feature is shuffled.

    Args:
        model: Fitted model with predict method
        X: Feature matrix
        y: Target values
        feature_names: Feature names
        n_repeats: Number of shuffles per feature
        scoring: Scoring metric ("mae", "accuracy")

    Returns:
        Dict mapping feature name to importance
    """
    logger.info(f"Calculating permutation importance for {len(feature_names)} features")

    # Baseline score
    predictions = model.predict(X)

    if scoring == "mae":
        baseline_score = np.mean(np.abs(y - predictions))
    else:
        baseline_score = np.mean(predictions == y)

    importance = {}

    for i, feature in enumerate(feature_names):
        scores = []

        for _ in range(n_repeats):
            X_permuted = X.copy()
            np.random.shuffle(X_permuted[:, i])

            perm_predictions = model.predict(X_permuted)

            if scoring == "mae":
                score = np.mean(np.abs(y - perm_predictions))
                # Higher MAE after permutation = more important
                importance_val = score - baseline_score
            else:
                score = np.mean(perm_predictions == y)
                # Lower accuracy after permutation = more important
                importance_val = baseline_score - score

            scores.append(importance_val)

        importance[feature] = float(np.mean(scores))

    return importance
