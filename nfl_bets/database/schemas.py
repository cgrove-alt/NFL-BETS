"""
Pydantic schemas for data validation and serialization.

Used for API responses, data transfer, and validation.
"""
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class BetType(str, Enum):
    """Types of bets."""

    SPREAD = "spread"
    MONEYLINE = "moneyline"
    TOTAL = "total"
    PLAYER_PROP = "player_prop"


class BetStatus(str, Enum):
    """Bet lifecycle status."""

    RECOMMENDED = "recommended"
    PLACED = "placed"
    SKIPPED = "skipped"
    WON = "won"
    LOST = "lost"
    PUSH = "push"
    VOID = "void"


class Urgency(str, Enum):
    """Alert urgency levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# =============================================================================
# GAME SCHEMAS
# =============================================================================
class GameBase(BaseModel):
    """Base schema for game data."""

    id: str
    season: int
    week: int
    game_type: str = "REG"
    game_date: datetime
    kickoff: datetime
    home_team: str
    away_team: str


class GameCreate(GameBase):
    """Schema for creating a new game."""

    stadium: Optional[str] = None
    roof: Optional[str] = None
    surface: Optional[str] = None
    opening_spread: Optional[float] = None
    opening_total: Optional[float] = None


class GameResponse(GameBase):
    """Schema for game response."""

    home_score: Optional[int] = None
    away_score: Optional[int] = None
    actual_spread: Optional[float] = None
    actual_total: Optional[float] = None
    closing_spread: Optional[float] = None
    closing_total: Optional[float] = None

    class Config:
        from_attributes = True


# =============================================================================
# ODDS SCHEMAS
# =============================================================================
class OddsOutcome(BaseModel):
    """Single odds outcome from a bookmaker."""

    name: str
    price: int  # American odds
    point: Optional[float] = None  # Spread/total point


class BookmakerOdds(BaseModel):
    """Odds from a single bookmaker."""

    key: str  # e.g., 'draftkings'
    title: str  # e.g., 'DraftKings'
    last_update: datetime
    markets: list["MarketOdds"]


class MarketOdds(BaseModel):
    """Odds for a single market type."""

    key: str  # e.g., 'spreads', 'h2h', 'totals'
    outcomes: list[OddsOutcome]


class GameOdds(BaseModel):
    """Full odds data for a game."""

    id: str
    sport_key: str
    commence_time: datetime
    home_team: str
    away_team: str
    bookmakers: list[BookmakerOdds]


# =============================================================================
# PREDICTION SCHEMAS
# =============================================================================
class SpreadPredictionBase(BaseModel):
    """Base schema for spread predictions."""

    game_id: str
    predicted_spread: float = Field(
        description="Predicted spread (positive = home favored)"
    )
    confidence_low: Optional[float] = None
    confidence_high: Optional[float] = None
    model_version: str


class SpreadPredictionCreate(SpreadPredictionBase):
    """Schema for creating a spread prediction."""

    home_win_probability: Optional[float] = None
    market_spread: Optional[float] = None
    market_odds: Optional[int] = None
    best_book: Optional[str] = None


class SpreadPredictionResponse(SpreadPredictionBase):
    """Schema for spread prediction response."""

    id: int
    created_at: datetime
    home_win_probability: Optional[float] = None
    edge: Optional[float] = None
    expected_value: Optional[float] = None
    is_value_bet: bool
    recommended_side: Optional[str] = None
    recommended_stake_pct: Optional[float] = None
    bet_placed: bool
    bet_won: Optional[bool] = None
    profit_loss: Optional[float] = None

    class Config:
        from_attributes = True


class PlayerPropPredictionBase(BaseModel):
    """Base schema for player prop predictions."""

    game_id: str
    player_id: str
    player_name: str
    team: str
    prop_type: str
    line: float
    predicted_value: float
    model_version: str


class PlayerPropPredictionCreate(PlayerPropPredictionBase):
    """Schema for creating a player prop prediction."""

    q10: Optional[float] = None
    q25: Optional[float] = None
    q75: Optional[float] = None
    q90: Optional[float] = None
    over_probability: Optional[float] = None
    over_odds: Optional[int] = None
    under_odds: Optional[int] = None
    best_book: Optional[str] = None


class PlayerPropPredictionResponse(PlayerPropPredictionBase):
    """Schema for player prop prediction response."""

    id: int
    created_at: datetime
    q10: Optional[float] = None
    q25: Optional[float] = None
    q75: Optional[float] = None
    q90: Optional[float] = None
    over_probability: Optional[float] = None
    edge_over: Optional[float] = None
    edge_under: Optional[float] = None
    is_value_bet: bool
    recommended_side: Optional[str] = None
    recommended_stake_pct: Optional[float] = None
    actual_value: Optional[float] = None
    bet_placed: bool
    bet_won: Optional[bool] = None
    profit_loss: Optional[float] = None

    class Config:
        from_attributes = True


# =============================================================================
# VALUE BET SCHEMAS
# =============================================================================
class ValueBet(BaseModel):
    """Identified value betting opportunity."""

    game_id: str
    description: str
    bet_type: BetType
    bookmaker: str

    # Probabilities
    model_probability: float = Field(ge=0, le=1)
    implied_probability: float = Field(ge=0, le=1)
    edge: float

    # Expected value
    expected_value: float

    # Kelly sizing
    kelly_fraction: float
    fractional_kelly: float
    recommended_stake_pct: float = Field(ge=0, le=1)

    # Odds
    american_odds: int
    decimal_odds: float

    # Metadata
    urgency: Urgency
    confidence_interval: Optional[tuple[float, float]] = None
    detected_at: datetime
    expires_at: Optional[datetime] = None

    @field_validator("edge")
    @classmethod
    def edge_must_be_reasonable(cls, v: float) -> float:
        if v > 0.5:
            raise ValueError("Edge > 50% is suspicious, likely an error")
        return v


class ArbitrageLeg(BaseModel):
    """One leg of an arbitrage bet."""

    bookmaker: str
    outcome: str
    american_odds: int
    decimal_odds: float
    stake: float
    potential_return: float


class ArbitrageOpportunity(BaseModel):
    """Complete arbitrage opportunity."""

    game_id: str
    game_description: str
    market_type: str

    profit_percentage: float
    guaranteed_profit: float
    total_stake: float

    legs: list[ArbitrageLeg]

    detected_at: datetime
    game_start: datetime
    time_to_start_minutes: int

    is_cross_market: bool = False
    max_stake_limited: bool = False


# =============================================================================
# BET TRACKING SCHEMAS
# =============================================================================
class BetCreate(BaseModel):
    """Schema for logging a bet."""

    bet_type: BetType
    prediction_id: Optional[int] = None
    game_id: Optional[str] = None
    description: str
    pick: str
    bookmaker: str
    odds: int
    stake: float
    model_probability: float
    implied_probability: float
    edge: float
    expected_value: float
    kelly_fraction: float


class BetResponse(BetCreate):
    """Schema for bet response."""

    id: int
    created_at: datetime
    status: BetStatus
    settled_at: Optional[datetime] = None
    profit_loss: Optional[float] = None
    bankroll_after: Optional[float] = None
    running_roi: Optional[float] = None

    class Config:
        from_attributes = True


class BetSettle(BaseModel):
    """Schema for settling a bet."""

    result: str = Field(pattern="^(win|loss|push|void)$")


# =============================================================================
# PERFORMANCE SCHEMAS
# =============================================================================
class PerformanceSummary(BaseModel):
    """Summary of betting performance."""

    period: str  # 'today', 'week', 'month', 'season', 'all'
    start_date: datetime
    end_date: datetime

    # Bet counts
    total_bets: int
    wins: int
    losses: int
    pushes: int
    pending: int

    # Win rate
    win_rate: Optional[float] = None

    # Financial
    total_staked: float
    total_profit_loss: float
    roi: Optional[float] = None

    # Edge metrics
    average_edge: Optional[float] = None
    average_ev: Optional[float] = None

    # By bet type
    by_bet_type: Optional[dict[str, "PerformanceByType"]] = None


class PerformanceByType(BaseModel):
    """Performance breakdown by bet type."""

    bet_type: str
    total_bets: int
    wins: int
    losses: int
    win_rate: Optional[float] = None
    total_staked: float
    profit_loss: float
    roi: Optional[float] = None


# =============================================================================
# BANKROLL SCHEMAS
# =============================================================================
class BankrollStatus(BaseModel):
    """Current bankroll status."""

    balance: float
    pending_bets: float
    available: float

    # Performance
    initial_balance: float
    total_profit_loss: float
    roi: Optional[float] = None

    # Risk metrics
    current_exposure: float
    max_single_bet: float


class BankrollUpdate(BaseModel):
    """Schema for updating bankroll."""

    amount: float
    change_type: str = Field(pattern="^(deposit|withdrawal)$")
    notes: Optional[str] = None


# =============================================================================
# NOTIFICATION SCHEMAS
# =============================================================================
class NotificationPriority(str, Enum):
    """Notification priority levels."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class Notification(BaseModel):
    """Notification payload."""

    title: str
    message: str
    priority: NotificationPriority
    category: str = "general"
    data: Optional[dict] = None


class AlertSettings(BaseModel):
    """User alert preferences."""

    console_enabled: bool = True
    desktop_enabled: bool = True
    discord_enabled: bool = False
    telegram_enabled: bool = False
    sms_enabled: bool = False

    min_edge_for_alert: float = 0.03
    min_priority: NotificationPriority = NotificationPriority.LOW


# =============================================================================
# API RESPONSE WRAPPERS
# =============================================================================
class PaginatedResponse(BaseModel):
    """Paginated response wrapper."""

    items: list
    total: int
    page: int
    page_size: int
    pages: int


class HealthCheck(BaseModel):
    """Health check response."""

    status: str = "healthy"
    database: bool = True
    odds_api: bool = True
    scheduler: bool = True
    last_odds_update: Optional[datetime] = None
    pending_value_bets: int = 0
