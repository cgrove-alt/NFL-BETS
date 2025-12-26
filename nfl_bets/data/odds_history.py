"""
Odds history database for tracking line movements and CLV.

Stores odds snapshots with timestamps to enable:
- CLV (Closing Line Value) tracking
- Line movement analysis
- Sharp money detection
- Historical odds comparison

Uses SQLite for local storage with efficient querying.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

from loguru import logger


class BetType(str, Enum):
    """Types of bets we track."""
    SPREAD = "spread"
    MONEYLINE = "moneyline"
    TOTAL = "total"
    PLAYER_PROP = "player_prop"


@dataclass
class OddsSnapshot:
    """A point-in-time snapshot of betting odds."""

    game_id: str
    timestamp: datetime
    bet_type: BetType
    sportsbook: str

    # For spreads
    home_spread: Optional[float] = None
    home_spread_odds: Optional[int] = None
    away_spread_odds: Optional[int] = None

    # For moneylines
    home_ml: Optional[int] = None
    away_ml: Optional[int] = None

    # For totals
    total_line: Optional[float] = None
    over_odds: Optional[int] = None
    under_odds: Optional[int] = None

    # For player props
    player_id: Optional[str] = None
    prop_type: Optional[str] = None
    prop_line: Optional[float] = None
    over_prop_odds: Optional[int] = None
    under_prop_odds: Optional[int] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "game_id": self.game_id,
            "timestamp": self.timestamp.isoformat(),
            "bet_type": self.bet_type.value,
            "sportsbook": self.sportsbook,
            "home_spread": self.home_spread,
            "home_spread_odds": self.home_spread_odds,
            "away_spread_odds": self.away_spread_odds,
            "home_ml": self.home_ml,
            "away_ml": self.away_ml,
            "total_line": self.total_line,
            "over_odds": self.over_odds,
            "under_odds": self.under_odds,
            "player_id": self.player_id,
            "prop_type": self.prop_type,
            "prop_line": self.prop_line,
            "over_prop_odds": self.over_prop_odds,
            "under_prop_odds": self.under_prop_odds,
        }


@dataclass
class CLVResult:
    """Closing Line Value calculation result."""

    game_id: str
    bet_type: BetType
    bet_line: float  # Line at time of bet
    closing_line: float  # Closing line
    clv_points: float  # Difference in points (for spreads/totals)
    clv_cents: float  # Difference in implied probability (for ML)
    is_positive: bool  # Did we get CLV?

    @property
    def clv_percentage(self) -> float:
        """CLV as percentage of line."""
        if self.bet_line == 0:
            return 0.0
        return abs(self.clv_points / self.bet_line) * 100


class OddsHistoryDB:
    """
    SQLite database for storing and querying odds history.

    Provides efficient storage and retrieval of odds snapshots
    for line movement analysis and CLV tracking.

    Example:
        >>> db = OddsHistoryDB("data/odds_history.db")
        >>> await db.initialize()
        >>> await db.store_snapshot(OddsSnapshot(
        ...     game_id="2024_01_KC_BAL",
        ...     timestamp=datetime.now(),
        ...     bet_type=BetType.SPREAD,
        ...     sportsbook="DraftKings",
        ...     home_spread=-3.5,
        ...     home_spread_odds=-110,
        ...     away_spread_odds=-110,
        ... ))
    """

    def __init__(self, db_path: str | Path = "data/odds_history.db"):
        """
        Initialize the odds history database.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None
        self.logger = logger.bind(component="odds_history")

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
        return self._conn

    async def initialize(self) -> None:
        """Create database tables if they don't exist."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Main odds snapshots table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS odds_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                bet_type TEXT NOT NULL,
                sportsbook TEXT NOT NULL,
                home_spread REAL,
                home_spread_odds INTEGER,
                away_spread_odds INTEGER,
                home_ml INTEGER,
                away_ml INTEGER,
                total_line REAL,
                over_odds INTEGER,
                under_odds INTEGER,
                player_id TEXT,
                prop_type TEXT,
                prop_line REAL,
                over_prop_odds INTEGER,
                under_prop_odds INTEGER,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Index for efficient queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_game_bet_type
            ON odds_snapshots (game_id, bet_type)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp
            ON odds_snapshots (timestamp)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_player_prop
            ON odds_snapshots (player_id, prop_type)
        """)

        # Tracked bets table for CLV calculation
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tracked_bets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id TEXT NOT NULL,
                bet_type TEXT NOT NULL,
                side TEXT NOT NULL,
                line REAL,
                odds INTEGER,
                stake REAL,
                placed_at TEXT NOT NULL,
                sportsbook TEXT,
                player_id TEXT,
                prop_type TEXT,
                closed INTEGER DEFAULT 0,
                closing_line REAL,
                result TEXT,
                pnl REAL
            )
        """)

        conn.commit()
        self.logger.info(f"Odds history database initialized at {self.db_path}")

    async def store_snapshot(self, snapshot: OddsSnapshot) -> int:
        """
        Store an odds snapshot.

        Args:
            snapshot: OddsSnapshot to store

        Returns:
            ID of inserted row
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO odds_snapshots (
                game_id, timestamp, bet_type, sportsbook,
                home_spread, home_spread_odds, away_spread_odds,
                home_ml, away_ml,
                total_line, over_odds, under_odds,
                player_id, prop_type, prop_line,
                over_prop_odds, under_prop_odds
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            snapshot.game_id,
            snapshot.timestamp.isoformat(),
            snapshot.bet_type.value,
            snapshot.sportsbook,
            snapshot.home_spread,
            snapshot.home_spread_odds,
            snapshot.away_spread_odds,
            snapshot.home_ml,
            snapshot.away_ml,
            snapshot.total_line,
            snapshot.over_odds,
            snapshot.under_odds,
            snapshot.player_id,
            snapshot.prop_type,
            snapshot.prop_line,
            snapshot.over_prop_odds,
            snapshot.under_prop_odds,
        ))

        conn.commit()
        return cursor.lastrowid

    async def store_snapshots(self, snapshots: list[OddsSnapshot]) -> int:
        """Store multiple snapshots efficiently."""
        conn = self._get_connection()
        cursor = conn.cursor()

        data = [
            (
                s.game_id, s.timestamp.isoformat(), s.bet_type.value, s.sportsbook,
                s.home_spread, s.home_spread_odds, s.away_spread_odds,
                s.home_ml, s.away_ml, s.total_line, s.over_odds, s.under_odds,
                s.player_id, s.prop_type, s.prop_line,
                s.over_prop_odds, s.under_prop_odds
            )
            for s in snapshots
        ]

        cursor.executemany("""
            INSERT INTO odds_snapshots (
                game_id, timestamp, bet_type, sportsbook,
                home_spread, home_spread_odds, away_spread_odds,
                home_ml, away_ml,
                total_line, over_odds, under_odds,
                player_id, prop_type, prop_line,
                over_prop_odds, under_prop_odds
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, data)

        conn.commit()
        return len(snapshots)

    async def get_opening_line(
        self,
        game_id: str,
        bet_type: BetType,
        player_id: Optional[str] = None,
        prop_type: Optional[str] = None,
    ) -> Optional[OddsSnapshot]:
        """Get the earliest (opening) line for a bet."""
        conn = self._get_connection()
        cursor = conn.cursor()

        query = """
            SELECT * FROM odds_snapshots
            WHERE game_id = ? AND bet_type = ?
        """
        params = [game_id, bet_type.value]

        if player_id:
            query += " AND player_id = ?"
            params.append(player_id)
        if prop_type:
            query += " AND prop_type = ?"
            params.append(prop_type)

        query += " ORDER BY timestamp ASC LIMIT 1"

        cursor.execute(query, params)
        row = cursor.fetchone()

        if row:
            return self._row_to_snapshot(row)
        return None

    async def get_closing_line(
        self,
        game_id: str,
        bet_type: BetType,
        player_id: Optional[str] = None,
        prop_type: Optional[str] = None,
    ) -> Optional[OddsSnapshot]:
        """Get the latest (closing) line for a bet."""
        conn = self._get_connection()
        cursor = conn.cursor()

        query = """
            SELECT * FROM odds_snapshots
            WHERE game_id = ? AND bet_type = ?
        """
        params = [game_id, bet_type.value]

        if player_id:
            query += " AND player_id = ?"
            params.append(player_id)
        if prop_type:
            query += " AND prop_type = ?"
            params.append(prop_type)

        query += " ORDER BY timestamp DESC LIMIT 1"

        cursor.execute(query, params)
        row = cursor.fetchone()

        if row:
            return self._row_to_snapshot(row)
        return None

    async def get_line_history(
        self,
        game_id: str,
        bet_type: BetType,
        player_id: Optional[str] = None,
        prop_type: Optional[str] = None,
        sportsbook: Optional[str] = None,
    ) -> list[OddsSnapshot]:
        """Get full line history for a bet."""
        conn = self._get_connection()
        cursor = conn.cursor()

        query = """
            SELECT * FROM odds_snapshots
            WHERE game_id = ? AND bet_type = ?
        """
        params = [game_id, bet_type.value]

        if player_id:
            query += " AND player_id = ?"
            params.append(player_id)
        if prop_type:
            query += " AND prop_type = ?"
            params.append(prop_type)
        if sportsbook:
            query += " AND sportsbook = ?"
            params.append(sportsbook)

        query += " ORDER BY timestamp ASC"

        cursor.execute(query, params)
        rows = cursor.fetchall()

        return [self._row_to_snapshot(row) for row in rows]

    async def calculate_line_movement(
        self,
        game_id: str,
        bet_type: BetType,
    ) -> dict[str, float]:
        """
        Calculate line movement for a game.

        Returns:
            Dict with opening line, closing line, and movement
        """
        opening = await self.get_opening_line(game_id, bet_type)
        closing = await self.get_closing_line(game_id, bet_type)

        if not opening or not closing:
            return {"opening": 0.0, "closing": 0.0, "movement": 0.0}

        if bet_type == BetType.SPREAD:
            opening_line = opening.home_spread or 0.0
            closing_line = closing.home_spread or 0.0
        elif bet_type == BetType.TOTAL:
            opening_line = opening.total_line or 0.0
            closing_line = closing.total_line or 0.0
        else:
            # For moneyline, use implied probability
            opening_line = self._american_to_prob(opening.home_ml or -110)
            closing_line = self._american_to_prob(closing.home_ml or -110)

        return {
            "opening": opening_line,
            "closing": closing_line,
            "movement": closing_line - opening_line,
        }

    async def track_bet(
        self,
        game_id: str,
        bet_type: BetType,
        side: str,  # "home", "away", "over", "under"
        line: float,
        odds: int,
        stake: float,
        sportsbook: Optional[str] = None,
        player_id: Optional[str] = None,
        prop_type: Optional[str] = None,
    ) -> int:
        """
        Track a placed bet for CLV calculation.

        Returns:
            ID of tracked bet
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO tracked_bets (
                game_id, bet_type, side, line, odds, stake,
                placed_at, sportsbook, player_id, prop_type
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            game_id, bet_type.value, side, line, odds, stake,
            datetime.now().isoformat(), sportsbook, player_id, prop_type
        ))

        conn.commit()
        self.logger.info(f"Tracked bet: {game_id} {bet_type.value} {side} {line}")
        return cursor.lastrowid

    async def calculate_clv(
        self,
        bet_id: int,
    ) -> Optional[CLVResult]:
        """
        Calculate CLV for a tracked bet.

        Args:
            bet_id: ID of tracked bet

        Returns:
            CLVResult or None if closing line not available
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        # Get the bet
        cursor.execute("SELECT * FROM tracked_bets WHERE id = ?", (bet_id,))
        bet = cursor.fetchone()

        if not bet:
            return None

        # Get closing line
        closing = await self.get_closing_line(
            bet["game_id"],
            BetType(bet["bet_type"]),
            bet["player_id"],
            bet["prop_type"],
        )

        if not closing:
            return None

        bet_line = bet["line"]
        bet_type = BetType(bet["bet_type"])

        # Calculate CLV based on bet type
        if bet_type == BetType.SPREAD:
            closing_line = closing.home_spread or bet_line
            if bet["side"] == "home":
                clv_points = bet_line - closing_line  # Getting better number
            else:
                clv_points = closing_line - bet_line
        elif bet_type == BetType.TOTAL:
            closing_line = closing.total_line or bet_line
            if bet["side"] == "over":
                clv_points = closing_line - bet_line  # Higher closing = CLV on over
            else:
                clv_points = bet_line - closing_line
        else:
            # Moneyline: compare implied probabilities
            closing_line = closing.home_ml or bet["odds"]
            bet_prob = self._american_to_prob(bet["odds"])
            close_prob = self._american_to_prob(int(closing_line))
            clv_points = (bet_prob - close_prob) * 100  # In cents

        clv_cents = clv_points * 10  # Rough conversion to cents

        # Update the bet with closing line
        cursor.execute("""
            UPDATE tracked_bets
            SET closing_line = ?, closed = 1
            WHERE id = ?
        """, (closing_line, bet_id))
        conn.commit()

        return CLVResult(
            game_id=bet["game_id"],
            bet_type=bet_type,
            bet_line=bet_line,
            closing_line=closing_line,
            clv_points=clv_points,
            clv_cents=clv_cents,
            is_positive=clv_points > 0,
        )

    async def get_clv_summary(self, days: int = 30) -> dict:
        """
        Get CLV summary for recent bets.

        Args:
            days: Number of days to include

        Returns:
            Dict with CLV statistics
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cutoff = datetime.now().isoformat()

        cursor.execute("""
            SELECT * FROM tracked_bets
            WHERE closed = 1
            ORDER BY placed_at DESC
            LIMIT 100
        """)

        bets = cursor.fetchall()

        if not bets:
            return {
                "total_bets": 0,
                "positive_clv_bets": 0,
                "avg_clv_points": 0.0,
                "clv_rate": 0.0,
            }

        total_clv = 0.0
        positive_clv = 0

        for bet in bets:
            if bet["line"] and bet["closing_line"]:
                clv = bet["line"] - bet["closing_line"]
                total_clv += clv
                if clv > 0:
                    positive_clv += 1

        return {
            "total_bets": len(bets),
            "positive_clv_bets": positive_clv,
            "avg_clv_points": total_clv / len(bets) if bets else 0.0,
            "clv_rate": positive_clv / len(bets) if bets else 0.0,
        }

    def _row_to_snapshot(self, row: sqlite3.Row) -> OddsSnapshot:
        """Convert database row to OddsSnapshot."""
        return OddsSnapshot(
            game_id=row["game_id"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            bet_type=BetType(row["bet_type"]),
            sportsbook=row["sportsbook"],
            home_spread=row["home_spread"],
            home_spread_odds=row["home_spread_odds"],
            away_spread_odds=row["away_spread_odds"],
            home_ml=row["home_ml"],
            away_ml=row["away_ml"],
            total_line=row["total_line"],
            over_odds=row["over_odds"],
            under_odds=row["under_odds"],
            player_id=row["player_id"],
            prop_type=row["prop_type"],
            prop_line=row["prop_line"],
            over_prop_odds=row["over_prop_odds"],
            under_prop_odds=row["under_prop_odds"],
        )

    @staticmethod
    def _american_to_prob(odds: int) -> float:
        """Convert American odds to implied probability."""
        if odds > 0:
            return 100.0 / (odds + 100.0)
        else:
            return abs(odds) / (abs(odds) + 100.0)

    async def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None


# Convenience function for quick CLV check
async def check_clv(
    db_path: str,
    game_id: str,
    bet_type: BetType,
    bet_line: float,
) -> Optional[float]:
    """
    Quick CLV check for a bet.

    Args:
        db_path: Path to odds database
        game_id: Game identifier
        bet_type: Type of bet
        bet_line: Line at which bet was placed

    Returns:
        CLV in points (positive = good)
    """
    db = OddsHistoryDB(db_path)
    await db.initialize()

    closing = await db.get_closing_line(game_id, bet_type)
    await db.close()

    if not closing:
        return None

    if bet_type == BetType.SPREAD:
        return bet_line - (closing.home_spread or bet_line)
    elif bet_type == BetType.TOTAL:
        return (closing.total_line or bet_line) - bet_line
    else:
        return 0.0
