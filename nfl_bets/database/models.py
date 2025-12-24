"""
SQLAlchemy ORM models for the NFL betting system.

Defines database schema for games, predictions, bets, and performance tracking.
"""
from datetime import datetime
from decimal import Decimal
from enum import Enum as PyEnum
from typing import Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all ORM models."""

    pass


class BetStatus(str, PyEnum):
    """Bet lifecycle status."""

    RECOMMENDED = "recommended"  # System recommended, user hasn't acted
    PLACED = "placed"  # User placed the bet
    SKIPPED = "skipped"  # User skipped the recommendation
    WON = "won"  # Bet won
    LOST = "lost"  # Bet lost
    PUSH = "push"  # Bet pushed (tie)
    VOID = "void"  # Bet voided


class BetType(str, PyEnum):
    """Types of bets."""

    SPREAD = "spread"
    MONEYLINE = "moneyline"
    TOTAL = "total"
    PLAYER_PROP = "player_prop"


class Game(Base):
    """NFL game information."""

    __tablename__ = "games"

    id: Mapped[str] = mapped_column(String(50), primary_key=True)  # nflverse game_id
    season: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    week: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    game_type: Mapped[str] = mapped_column(
        String(10), default="REG"
    )  # REG, POST, WC, DIV, CON, SB
    game_date: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    kickoff: Mapped[datetime] = mapped_column(DateTime, nullable=False)

    home_team: Mapped[str] = mapped_column(String(5), nullable=False, index=True)
    away_team: Mapped[str] = mapped_column(String(5), nullable=False, index=True)

    # Final scores (null until game completes)
    home_score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    away_score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Actual results for tracking
    actual_spread: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )  # home - away
    actual_total: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Game context
    stadium: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    roof: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)  # dome, open
    surface: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)

    # Vegas lines at kickoff
    opening_spread: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    closing_spread: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    opening_total: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    closing_total: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Metadata
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    # Relationships
    spread_predictions: Mapped[list["SpreadPrediction"]] = relationship(
        back_populates="game"
    )
    prop_predictions: Mapped[list["PlayerPropPrediction"]] = relationship(
        back_populates="game"
    )

    __table_args__ = (
        Index("ix_games_season_week", "season", "week"),
        Index("ix_games_date", "game_date"),
    )


class SpreadPrediction(Base):
    """Model predictions for game spreads."""

    __tablename__ = "spread_predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    game_id: Mapped[str] = mapped_column(
        String(50), ForeignKey("games.id"), nullable=False, index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )

    # Model prediction (positive = home favored)
    predicted_spread: Mapped[float] = mapped_column(Float, nullable=False)
    confidence_low: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    confidence_high: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    model_version: Mapped[str] = mapped_column(String(50), nullable=False)

    # Home team win probability
    home_win_probability: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Market data at prediction time
    market_spread: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    market_odds: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True
    )  # American odds
    best_book: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    # Value detection
    edge: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    expected_value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    is_value_bet: Mapped[bool] = mapped_column(Boolean, default=False)
    recommended_side: Mapped[Optional[str]] = mapped_column(
        String(10), nullable=True
    )  # 'home', 'away'
    recommended_stake_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Outcome tracking
    bet_placed: Mapped[bool] = mapped_column(Boolean, default=False)
    bet_won: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    profit_loss: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Relationship
    game: Mapped["Game"] = relationship(back_populates="spread_predictions")

    __table_args__ = (Index("ix_spread_predictions_created", "created_at"),)


class PlayerPropPrediction(Base):
    """Model predictions for player props."""

    __tablename__ = "player_prop_predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    game_id: Mapped[str] = mapped_column(
        String(50), ForeignKey("games.id"), nullable=False, index=True
    )
    player_id: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    player_name: Mapped[str] = mapped_column(String(100), nullable=False)
    team: Mapped[str] = mapped_column(String(5), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )

    # Prop details
    prop_type: Mapped[str] = mapped_column(
        String(50), nullable=False, index=True
    )  # passing_yards, rushing_yards, etc.
    line: Mapped[float] = mapped_column(Float, nullable=False)

    # Model prediction
    predicted_value: Mapped[float] = mapped_column(Float, nullable=False)
    q10: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # 10th percentile
    q25: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # 25th percentile
    q75: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # 75th percentile
    q90: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # 90th percentile
    over_probability: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    model_version: Mapped[str] = mapped_column(String(50), nullable=False)

    # Market data
    over_odds: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    under_odds: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    best_book: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    # Value detection
    edge_over: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    edge_under: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    is_value_bet: Mapped[bool] = mapped_column(Boolean, default=False)
    recommended_side: Mapped[Optional[str]] = mapped_column(
        String(10), nullable=True
    )  # 'over', 'under'
    recommended_stake_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Outcome tracking
    actual_value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    bet_placed: Mapped[bool] = mapped_column(Boolean, default=False)
    bet_won: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    profit_loss: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Relationship
    game: Mapped["Game"] = relationship(back_populates="prop_predictions")

    __table_args__ = (
        Index("ix_prop_predictions_player", "player_id"),
        Index("ix_prop_predictions_type", "prop_type"),
    )


class BetHistory(Base):
    """Track all betting activity for performance analysis."""

    __tablename__ = "bet_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False, index=True
    )

    # Bet identification
    bet_type: Mapped[str] = mapped_column(Enum(BetType), nullable=False)
    prediction_id: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True
    )  # Links to spread or prop prediction
    game_id: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    # Bet details
    description: Mapped[str] = mapped_column(String(200), nullable=False)
    pick: Mapped[str] = mapped_column(String(100), nullable=False)
    bookmaker: Mapped[str] = mapped_column(String(50), nullable=False)
    odds: Mapped[int] = mapped_column(Integer, nullable=False)  # American odds
    stake: Mapped[float] = mapped_column(Float, nullable=False)

    # Edge metrics at bet time
    model_probability: Mapped[float] = mapped_column(Float, nullable=False)
    implied_probability: Mapped[float] = mapped_column(Float, nullable=False)
    edge: Mapped[float] = mapped_column(Float, nullable=False)
    expected_value: Mapped[float] = mapped_column(Float, nullable=False)
    kelly_fraction: Mapped[float] = mapped_column(Float, nullable=False)

    # Status and outcome
    status: Mapped[str] = mapped_column(
        Enum(BetStatus), default=BetStatus.RECOMMENDED, nullable=False
    )
    settled_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    profit_loss: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Running totals (denormalized for quick queries)
    bankroll_after: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    running_roi: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    __table_args__ = (
        Index("ix_bet_history_status", "status"),
        Index("ix_bet_history_type", "bet_type"),
    )


class OddsSnapshot(Base):
    """Historical odds snapshots for line movement tracking."""

    __tablename__ = "odds_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    captured_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False, index=True
    )

    game_id: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    bookmaker: Mapped[str] = mapped_column(String(50), nullable=False)
    market_type: Mapped[str] = mapped_column(
        String(20), nullable=False
    )  # spread, h2h, totals

    # Odds data
    outcome_name: Mapped[str] = mapped_column(String(100), nullable=False)
    american_odds: Mapped[int] = mapped_column(Integer, nullable=False)
    point: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # Spread/total

    __table_args__ = (
        Index("ix_odds_snapshots_game", "game_id", "market_type"),
        Index("ix_odds_snapshots_time", "captured_at"),
    )


class ModelPerformance(Base):
    """Track model performance metrics over time."""

    __tablename__ = "model_performance"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    date: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    model_type: Mapped[str] = mapped_column(
        String(50), nullable=False, index=True
    )  # spread, passing_yards, etc.
    model_version: Mapped[str] = mapped_column(String(50), nullable=False)

    # Accuracy metrics
    num_predictions: Mapped[int] = mapped_column(Integer, nullable=False)
    mae: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    rmse: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # For classification metrics (ATS, over/under)
    accuracy: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Betting performance
    num_value_bets: Mapped[int] = mapped_column(Integer, default=0)
    num_bets_placed: Mapped[int] = mapped_column(Integer, default=0)
    win_rate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    roi: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    total_profit_loss: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Calibration
    calibration_error: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    brier_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)


class BankrollHistory(Base):
    """Track bankroll changes over time."""

    __tablename__ = "bankroll_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False, index=True
    )

    balance: Mapped[float] = mapped_column(Float, nullable=False)
    pending_bets: Mapped[float] = mapped_column(Float, default=0)
    available: Mapped[float] = mapped_column(Float, nullable=False)

    # Change details
    change_amount: Mapped[float] = mapped_column(Float, nullable=False)
    change_type: Mapped[str] = mapped_column(
        String(20), nullable=False
    )  # deposit, withdrawal, bet, win, loss
    bet_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


def init_db(database_url: str) -> None:
    """
    Initialize the database with all tables.

    Args:
        database_url: SQLAlchemy database URL
    """
    engine = create_engine(database_url)
    Base.metadata.create_all(engine)


def get_engine(database_url: str):
    """
    Get SQLAlchemy engine for database operations.

    Args:
        database_url: SQLAlchemy database URL

    Returns:
        SQLAlchemy Engine
    """
    return create_engine(database_url)
