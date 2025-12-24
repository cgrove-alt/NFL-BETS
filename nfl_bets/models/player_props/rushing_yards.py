"""
Rushing yards prop model for RB predictions.

Specialized model for predicting running back rushing yards
with emphasis on carry volume and game script considerations.
"""
from typing import Optional

from .base_prop import BasePropModel


class RushingYardsModel(BasePropModel):
    """
    Model for predicting RB rushing yards.

    Key predictive features:
    - Carries (volume is king for rushing)
    - Yards per carry (efficiency)
    - Team implied score (game script matters)
    - Opponent rush defense EPA allowed
    - O-line health (crucial for rushing)

    Game script note: Teams with leads run more, so positive game script
    (team expected to win) is bullish for rushing props.

    Example:
        >>> model = RushingYardsModel()
        >>> model.train(X_train, y_train)
        >>> prediction = model.predict_player(
        ...     features, player_id="00-0037077",
        ...     player_name="Derrick Henry", game_id="2024_01_BAL_KC",
        ...     team="BAL", opponent="KC", line=85.5
        ... )
    """

    MODEL_TYPE = "prop_quantile"
    VERSION = "1.0.0"

    prop_type = "rushing_yards"
    position_filter = ["RB", "FB"]

    # Feature weights - emphasize volume and game script
    FEATURE_WEIGHTS = {
        # Volume features (most important)
        "carries_5g": 1.6,
        "carries_10g": 1.4,
        "carries_3g": 1.5,
        "snap_share_5g": 1.2,

        # Efficiency features
        "yards_per_carry_5g": 1.4,
        "yards_per_carry_10g": 1.2,
        "rushing_epa_5g": 1.2,

        # Matchup features (crucial)
        "opp_rush_epa_allowed": 1.5,
        "opp_rush_epa_allowed_5g": 1.4,

        # Game script (very important for RBs)
        "team_implied_score": 1.4,  # Higher = more rushing opportunity
        "game_total": 1.0,

        # Spread context
        "spread_favorite": 1.2,  # Favorites run more

        # O-line health (crucial for running game)
        "oline_health": 1.3,
        "position_group_health": 1.1,

        # Trend features
        "carries_trend": 1.2,
        "usage_trend": 1.1,

        # Receiving work (can impact rushing usage)
        "targets_5g": 0.9,
        "target_share_5g": 0.9,

        # Player health
        "player_injury_status": 1.1,

        # Team run tendency
        "team_run_rate_5g": 1.2,
    }

    # Model parameters tuned for rushing yards
    MODEL_PARAMS = {
        "n_estimators": 400,
        "max_depth": 5,
        "learning_rate": 0.04,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 8,  # RBs have more data points
        "reg_alpha": 0.15,
        "reg_lambda": 1.2,
        "random_state": 42,
    }

    def __init__(
        self,
        quantiles: Optional[list[float]] = None,
        model_params: Optional[dict] = None,
        use_calibration: bool = True,
    ):
        """Initialize the rushing yards model."""
        params = {**self.MODEL_PARAMS, **(model_params or {})}
        super().__init__(
            quantiles=quantiles,
            model_params=params,
            use_calibration=use_calibration,
        )

    @classmethod
    def get_relevant_features(cls) -> list[str]:
        """Get list of features most relevant for rushing yards."""
        return [
            # Volume
            "carries_5g",
            "carries_10g",
            "carries_3g",
            "snap_share_5g",
            # Efficiency
            "yards_per_carry_5g",
            "rushing_epa_5g",
            # Matchup
            "opp_rush_epa_allowed",
            # Game script
            "team_implied_score",
            "game_total",
            # Team context
            "oline_health",
            "team_run_rate_5g",
            # Trend
            "carries_trend",
            "usage_trend",
            # Health
            "player_injury_status",
        ]
