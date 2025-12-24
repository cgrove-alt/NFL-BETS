"""
Passing yards prop model for QB predictions.

Specialized model for predicting quarterback passing yards
with emphasis on volume indicators (attempts) and matchup quality.
"""
from typing import Optional

from .base_prop import BasePropModel


class PassingYardsModel(BasePropModel):
    """
    Model for predicting QB passing yards.

    Key predictive features:
    - Pass attempts (volume is king for passing yards)
    - Yards per attempt (efficiency)
    - CPOE (Completion Percentage Over Expected)
    - Opponent pass defense EPA allowed
    - Game total (high totals = more passing)
    - Indoor/dome games (typically higher passing)

    Example:
        >>> model = PassingYardsModel()
        >>> model.train(X_train, y_train)
        >>> prediction = model.predict_player(
        ...     features, player_id="00-0036971",
        ...     player_name="Patrick Mahomes", game_id="2024_01_KC_BAL",
        ...     team="KC", opponent="BAL", line=275.5
        ... )
        >>> print(f"Over probability: {prediction.over_prob:.1%}")
    """

    MODEL_TYPE = "prop_quantile"
    VERSION = "1.0.0"

    prop_type = "passing_yards"
    position_filter = ["QB"]

    # Feature weights - emphasize volume and matchup
    FEATURE_WEIGHTS = {
        # Volume features (most important)
        "pass_attempts_5g": 1.5,
        "pass_attempts_10g": 1.3,
        "pass_attempts_3g": 1.4,

        # Efficiency features
        "yards_per_attempt_5g": 1.3,
        "yards_per_attempt_10g": 1.2,
        "cpoe_5g": 1.2,
        "completion_rate_5g": 1.1,

        # EPA features
        "passing_epa_5g": 1.2,
        "passing_epa_10g": 1.1,

        # Matchup features (crucial for passing)
        "opp_pass_epa_allowed": 1.4,
        "opp_pass_epa_allowed_5g": 1.3,

        # Game context (high totals = more passing)
        "game_total": 1.3,
        "team_implied_score": 1.2,

        # Venue (indoor typically higher passing)
        "is_indoor": 1.1,
        "is_outdoor": 0.9,

        # Trend features
        "pass_attempts_trend": 1.1,
        "target_trend": 1.0,

        # Player health
        "player_injury_status": 1.0,
        "position_group_health": 0.9,

        # Default weight for unlisted features
    }

    # Model parameters tuned for passing yards
    MODEL_PARAMS = {
        "n_estimators": 400,
        "max_depth": 5,
        "learning_rate": 0.04,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 10,  # QBs have less data than skill players
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
    }

    def __init__(
        self,
        quantiles: Optional[list[float]] = None,
        model_params: Optional[dict] = None,
        use_calibration: bool = True,
    ):
        """Initialize the passing yards model."""
        # Merge custom params with defaults
        params = {**self.MODEL_PARAMS, **(model_params or {})}
        super().__init__(
            quantiles=quantiles,
            model_params=params,
            use_calibration=use_calibration,
        )

    @classmethod
    def get_relevant_features(cls) -> list[str]:
        """Get list of features most relevant for passing yards."""
        return [
            # Volume
            "pass_attempts_5g",
            "pass_attempts_10g",
            "pass_attempts_3g",
            # Efficiency
            "yards_per_attempt_5g",
            "completion_rate_5g",
            "cpoe_5g",
            # EPA
            "passing_epa_5g",
            # Matchup
            "opp_pass_epa_allowed",
            # Context
            "game_total",
            "team_implied_score",
            "is_indoor",
            # Trend
            "pass_attempts_trend",
            # Health
            "player_injury_status",
        ]
