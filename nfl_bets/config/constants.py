"""
Constants and feature definitions for the NFL betting system.

Contains feature lists, column definitions, team mappings, and thresholds.
"""
from enum import Enum
from typing import Final


# =============================================================================
# BET TYPES
# =============================================================================
class BetType(str, Enum):
    """Supported bet types."""

    SPREAD = "spread"
    MONEYLINE = "moneyline"
    TOTAL = "total"
    PLAYER_PROP = "player_prop"


class PropType(str, Enum):
    """Supported player prop types."""

    PASSING_YARDS = "player_pass_yards"
    PASSING_TDS = "player_pass_tds"
    PASSING_ATTEMPTS = "player_pass_attempts"
    PASSING_COMPLETIONS = "player_pass_completions"
    INTERCEPTIONS = "player_pass_ints"
    RUSHING_YARDS = "player_rush_yards"
    RUSHING_ATTEMPTS = "player_rush_attempts"
    RUSHING_TDS = "player_rush_tds"
    RECEIVING_YARDS = "player_rec_yards"
    RECEPTIONS = "player_rec"
    RECEIVING_TDS = "player_rec_tds"
    ANYTIME_TD = "player_anytime_td"
    FIRST_TD = "player_first_td"


class Urgency(str, Enum):
    """Alert urgency levels based on edge magnitude."""

    LOW = "low"  # 3-5% edge
    MEDIUM = "medium"  # 5-8% edge
    HIGH = "high"  # 8-12% edge
    CRITICAL = "critical"  # >12% edge


# =============================================================================
# EDGE THRESHOLDS
# =============================================================================
EDGE_THRESHOLDS: Final[dict[Urgency, tuple[float, float]]] = {
    Urgency.LOW: (0.03, 0.05),
    Urgency.MEDIUM: (0.05, 0.08),
    Urgency.HIGH: (0.08, 0.12),
    Urgency.CRITICAL: (0.12, 1.0),
}


# =============================================================================
# NFL TEAM MAPPINGS
# =============================================================================
# Standard team abbreviations used by nflfastR
NFL_TEAMS: Final[list[str]] = [
    "ARI",
    "ATL",
    "BAL",
    "BUF",
    "CAR",
    "CHI",
    "CIN",
    "CLE",
    "DAL",
    "DEN",
    "DET",
    "GB",
    "HOU",
    "IND",
    "JAX",
    "KC",
    "LA",
    "LAC",
    "LV",
    "MIA",
    "MIN",
    "NE",
    "NO",
    "NYG",
    "NYJ",
    "PHI",
    "PIT",
    "SEA",
    "SF",
    "TB",
    "TEN",
    "WAS",
]

# Mapping from various name formats to standard abbreviations
TEAM_NAME_MAPPING: Final[dict[str, str]] = {
    # Full names
    "Arizona Cardinals": "ARI",
    "Atlanta Falcons": "ATL",
    "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR",
    "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN",
    "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN",
    "Detroit Lions": "DET",
    "Green Bay Packers": "GB",
    "Houston Texans": "HOU",
    "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC",
    "Los Angeles Rams": "LA",
    "Los Angeles Chargers": "LAC",
    "Las Vegas Raiders": "LV",
    "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN",
    "New England Patriots": "NE",
    "New Orleans Saints": "NO",
    "New York Giants": "NYG",
    "New York Jets": "NYJ",
    "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT",
    "Seattle Seahawks": "SEA",
    "San Francisco 49ers": "SF",
    "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN",
    "Washington Commanders": "WAS",
    # Short names (used by some APIs)
    "Cardinals": "ARI",
    "Falcons": "ATL",
    "Ravens": "BAL",
    "Bills": "BUF",
    "Panthers": "CAR",
    "Bears": "CHI",
    "Bengals": "CIN",
    "Browns": "CLE",
    "Cowboys": "DAL",
    "Broncos": "DEN",
    "Lions": "DET",
    "Packers": "GB",
    "Texans": "HOU",
    "Colts": "IND",
    "Jaguars": "JAX",
    "Chiefs": "KC",
    "Rams": "LA",
    "Chargers": "LAC",
    "Raiders": "LV",
    "Dolphins": "MIA",
    "Vikings": "MIN",
    "Patriots": "NE",
    "Saints": "NO",
    "Giants": "NYG",
    "Jets": "NYJ",
    "Eagles": "PHI",
    "Steelers": "PIT",
    "Seahawks": "SEA",
    "49ers": "SF",
    "Buccaneers": "TB",
    "Titans": "TEN",
    "Commanders": "WAS",
    # Historical names
    "Washington Football Team": "WAS",
    "Washington Redskins": "WAS",
    "Oakland Raiders": "LV",
    "San Diego Chargers": "LAC",
    "St. Louis Rams": "LA",
}


# =============================================================================
# FEATURE DEFINITIONS - TEAM LEVEL
# =============================================================================
TEAM_EPA_FEATURES: Final[list[str]] = [
    "epa_per_play",
    "epa_per_pass",
    "epa_per_rush",
    "success_rate",
    "explosive_play_rate",
    "negative_play_rate",
]

TEAM_PASSING_FEATURES: Final[list[str]] = [
    "cpoe",
    "air_yards_per_attempt",
    "yac_per_completion",
    "sack_rate",
    "interception_rate",
    "scramble_rate",
    "play_action_rate",
    "dropback_epa",
]

TEAM_RUSHING_FEATURES: Final[list[str]] = [
    "yards_per_carry",
    "yards_before_contact",
    "stuffed_rate",
    "first_down_rate_rush",
    "epa_per_rush",
]

TEAM_SITUATIONAL_FEATURES: Final[list[str]] = [
    "red_zone_td_rate",
    "third_down_conv_rate",
    "fourth_down_conv_rate",
    "first_down_epa",
    "early_down_pass_rate",
    "two_minute_epa",
    "fourth_quarter_epa",
    "garbage_time_play_rate",
]

TEAM_DEFENSIVE_FEATURES: Final[list[str]] = [
    "epa_allowed_per_play",
    "epa_allowed_per_pass",
    "epa_allowed_per_rush",
    "pressure_rate",
    "sack_rate_defense",
    "turnover_rate_forced",
    "explosive_play_allowed_rate",
    "red_zone_td_allowed_rate",
    "third_down_stop_rate",
]


# =============================================================================
# FEATURE DEFINITIONS - PLAYER LEVEL
# =============================================================================
PLAYER_VOLUME_FEATURES: Final[list[str]] = [
    "targets",
    "target_share",
    "air_yards_share",
    "routes_run",
    "snap_share",
    "red_zone_targets",
    "goal_line_carries",
]

PLAYER_EFFICIENCY_FEATURES: Final[list[str]] = [
    "catch_rate",
    "yards_per_route_run",
    "yards_per_reception",
    "contested_catch_rate",
    "separation_score",
    "yards_per_carry",
    "yards_after_contact",
    "broken_tackle_rate",
]

PLAYER_PFF_FEATURES: Final[list[str]] = [
    "pff_receiving_grade",
    "pff_route_running_grade",
    "pff_rushing_grade",
    "pff_pass_blocking_grade",
    "pff_run_blocking_grade",
    "pff_passing_grade",
    "pff_coverage_grade",
]


# =============================================================================
# ROLLING WINDOW CONFIGURATIONS
# =============================================================================
ROLLING_WINDOWS: Final[list[int]] = [3, 5, 10, 17]  # Games lookback
ROLLING_WINDOW_LABELS: Final[dict[int, str]] = {
    3: "3g",  # Last 3 games
    5: "5g",  # Last 5 games
    10: "10g",  # Last 10 games
    17: "season",  # Full season
}


# =============================================================================
# NFLFASTR COLUMN SUBSETS
# =============================================================================
# Key columns to load from play-by-play data (reduces memory usage)
PBP_COLUMNS_CORE: Final[list[str]] = [
    "game_id",
    "play_id",
    "season",
    "week",
    "game_date",
    "posteam",
    "defteam",
    "home_team",
    "away_team",
    "play_type",
    "desc",
]

PBP_COLUMNS_PLAY_DETAILS: Final[list[str]] = [
    "down",
    "ydstogo",
    "yardline_100",
    "yards_gained",
    "touchdown",
    "interception",
    "fumble",
    "fumble_lost",
    "sack",
    "complete_pass",
    "incomplete_pass",
    "pass_attempt",
    "rush_attempt",
]

PBP_COLUMNS_EPA: Final[list[str]] = [
    "epa",
    "air_epa",
    "yac_epa",
    "qb_epa",
    "wp",
    "wpa",
    "vegas_wpa",
]

PBP_COLUMNS_PASSING: Final[list[str]] = [
    "passer_id",
    "passer_player_name",
    "receiver_id",
    "receiver_player_name",
    "air_yards",
    "yards_after_catch",
    "cpoe",
    "pass_location",
]

PBP_COLUMNS_RUSHING: Final[list[str]] = [
    "rusher_id",
    "rusher_player_name",
    "rushing_yards",
]

PBP_COLUMNS_SITUATIONAL: Final[list[str]] = [
    "score_differential",
    "half_seconds_remaining",
    "game_seconds_remaining",
    "goal_to_go",
    "drive",
    "series",
]

# Combined list of all columns to load
PBP_COLUMNS_ALL: Final[list[str]] = (
    PBP_COLUMNS_CORE
    + PBP_COLUMNS_PLAY_DETAILS
    + PBP_COLUMNS_EPA
    + PBP_COLUMNS_PASSING
    + PBP_COLUMNS_RUSHING
    + PBP_COLUMNS_SITUATIONAL
)


# =============================================================================
# BOOKMAKER MAPPINGS
# =============================================================================
BOOKMAKER_DISPLAY_NAMES: Final[dict[str, str]] = {
    "draftkings": "DraftKings",
    "fanduel": "FanDuel",
    "betmgm": "BetMGM",
    "caesars": "Caesars",
    "pointsbetus": "PointsBet",
    "barstool": "Barstool",
    "unibet_us": "Unibet",
    "betrivers": "BetRivers",
    "bovada": "Bovada",
    "mybookieag": "MyBookie",
    "betonlineag": "BetOnline",
    "lowvig": "LowVig",
    "pinnacle": "Pinnacle",
}


# =============================================================================
# MODEL CONFIGURATIONS
# =============================================================================
SPREAD_MODEL_FEATURES: Final[list[str]] = [
    # Team offensive EPA (rolling)
    "home_epa_per_play_5g",
    "away_epa_per_play_5g",
    "home_epa_per_pass_5g",
    "away_epa_per_pass_5g",
    "home_epa_per_rush_5g",
    "away_epa_per_rush_5g",
    # Team defensive EPA (rolling)
    "home_epa_allowed_per_play_5g",
    "away_epa_allowed_per_play_5g",
    # Passing metrics
    "home_cpoe_5g",
    "away_cpoe_5g",
    # Situational
    "home_red_zone_td_rate_5g",
    "away_red_zone_td_rate_5g",
    "home_third_down_conv_rate_5g",
    "away_third_down_conv_rate_5g",
    # DVOA (if available)
    "home_total_dvoa",
    "away_total_dvoa",
    "home_offense_dvoa",
    "away_offense_dvoa",
    "home_defense_dvoa",
    "away_defense_dvoa",
    # Injury impact
    "home_team_health_score",
    "away_team_health_score",
    # Context
    "home_rest_days",
    "away_rest_days",
    "is_divisional",
    "is_primetime",
    # Vegas context
    "opening_spread",
    "implied_total",
]

# Default XGBoost parameters for spread model
XGBOOST_SPREAD_PARAMS: Final[dict] = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "min_child_weight": 3,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "early_stopping_rounds": 50,
    "random_state": 42,
}


# =============================================================================
# DATA CACHE SETTINGS
# =============================================================================
CACHE_TTL_SECONDS: Final[dict[str, int]] = {
    "odds": 300,  # 5 minutes
    "pbp": 86400,  # 24 hours
    "player_stats": 86400,  # 24 hours
    "roster": 86400,  # 24 hours
    "schedule": 3600,  # 1 hour
    "predictions": 1800,  # 30 minutes
}
