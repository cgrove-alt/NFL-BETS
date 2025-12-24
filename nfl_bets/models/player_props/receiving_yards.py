"""
Receiving yards prop model for WR/TE predictions.

Specialized model for predicting wide receiver and tight end
receiving yards with emphasis on target share and matchup quality.
"""
from typing import Optional

from .base_prop import BasePropModel


class ReceivingYardsModel(BasePropModel):
    """
    Model for predicting WR/TE receiving yards.

    Key predictive features:
    - Targets (volume drives receiving production)
    - Target share (share of team's passing attack)
    - Yards per target (efficiency)
    - Opponent pass defense EPA allowed
    - QB quality (better QBs = more opportunity)
    - Game total (high totals = more passing volume)

    Example:
        >>> model = ReceivingYardsModel()
        >>> model.train(X_train, y_train)
        >>> prediction = model.predict_player(
        ...     features, player_id="00-0036212",
        ...     player_name="Ja'Marr Chase", game_id="2024_01_CIN_CLE",
        ...     team="CIN", opponent="CLE", line=75.5
        ... )
    """

    MODEL_TYPE = "prop_quantile"
    VERSION = "1.0.0"

    prop_type = "receiving_yards"
    position_filter = ["WR", "TE"]

    # Feature weights - emphasize target volume and matchup
    FEATURE_WEIGHTS = {
        # Volume features (most important for WRs)
        "targets_5g": 1.6,
        "targets_10g": 1.4,
        "targets_3g": 1.5,
        "target_share_5g": 1.5,
        "target_share_10g": 1.3,

        # Route running (more routes = more opportunity)
        "routes_run_5g": 1.3,
        "route_participation_5g": 1.2,
        "snap_share_5g": 1.2,

        # Air yards (big play potential)
        "air_yards_5g": 1.3,
        "air_yards_share_5g": 1.2,
        "adot_5g": 1.1,  # Average depth of target

        # Efficiency features
        "yards_per_target_5g": 1.3,
        "yards_per_reception_5g": 1.2,
        "catch_rate_5g": 1.1,
        "receiving_epa_5g": 1.2,

        # Matchup features
        "opp_pass_epa_allowed": 1.4,
        "opp_pass_epa_allowed_5g": 1.3,

        # Game context
        "game_total": 1.3,
        "team_implied_score": 1.1,

        # Spread context (trailing teams pass more)
        "spread_underdog": 1.1,

        # QB quality indicator
        "team_passing_epa_5g": 1.2,

        # Trend features
        "target_trend": 1.2,
        "usage_trend": 1.1,

        # Injury considerations
        "player_injury_status": 1.0,
        "position_group_health": 1.0,

        # WR room competition
        "team_wr1_targets": 0.9,  # If this player IS WR1, helps them
    }

    # Model parameters tuned for receiving yards
    MODEL_PARAMS = {
        "n_estimators": 400,
        "max_depth": 6,
        "learning_rate": 0.04,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 8,
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
        """Initialize the receiving yards model."""
        params = {**self.MODEL_PARAMS, **(model_params or {})}
        super().__init__(
            quantiles=quantiles,
            model_params=params,
            use_calibration=use_calibration,
        )

    @classmethod
    def get_relevant_features(cls) -> list[str]:
        """Get list of features most relevant for receiving yards."""
        return [
            # Volume
            "targets_5g",
            "targets_10g",
            "target_share_5g",
            "routes_run_5g",
            "snap_share_5g",
            # Air yards
            "air_yards_5g",
            "air_yards_share_5g",
            "adot_5g",
            # Efficiency
            "yards_per_target_5g",
            "catch_rate_5g",
            "receiving_epa_5g",
            # Matchup
            "opp_pass_epa_allowed",
            # Context
            "game_total",
            "team_implied_score",
            "team_passing_epa_5g",
            # Trend
            "target_trend",
            # Health
            "player_injury_status",
        ]


class ReceptionsModel(BasePropModel):
    """
    Model for predicting WR/TE receptions.

    Similar to receiving yards but even more focused on volume
    and catch rate rather than efficiency.
    """

    MODEL_TYPE = "prop_quantile"
    VERSION = "1.0.0"

    prop_type = "receptions"
    position_filter = ["WR", "TE"]

    FEATURE_WEIGHTS = {
        # Volume features (dominant for receptions)
        "targets_5g": 1.7,
        "targets_10g": 1.5,
        "target_share_5g": 1.5,

        # Catch rate (crucial for receptions)
        "catch_rate_5g": 1.5,
        "catch_rate_10g": 1.3,
        "receptions_5g": 1.4,

        # Route running
        "routes_run_5g": 1.3,
        "snap_share_5g": 1.2,

        # ADOT (lower = more short targets = more catches)
        "adot_5g": 0.9,

        # Matchup
        "opp_pass_epa_allowed": 1.2,

        # Context
        "game_total": 1.2,
        "team_implied_score": 1.0,

        # Trend
        "target_trend": 1.2,

        # Health
        "player_injury_status": 1.0,
    }

    MODEL_PARAMS = {
        "n_estimators": 350,
        "max_depth": 5,
        "learning_rate": 0.04,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 10,
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
        """Initialize the receptions model."""
        params = {**self.MODEL_PARAMS, **(model_params or {})}
        super().__init__(
            quantiles=quantiles,
            model_params=params,
            use_calibration=use_calibration,
        )
