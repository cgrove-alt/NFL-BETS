"""
Bankroll management and bet tracking.

Provides balance tracking, bet history, exposure monitoring,
and drawdown detection for responsible betting.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional
import json
from pathlib import Path


class BetStatus(str, Enum):
    """Status of a placed bet."""

    PENDING = "pending"
    WON = "won"
    LOST = "lost"
    PUSH = "push"
    VOIDED = "voided"
    CASHED_OUT = "cashed_out"


class BetType(str, Enum):
    """Type of bet placed."""

    SPREAD = "spread"
    MONEYLINE = "moneyline"
    TOTAL = "total"
    PLAYER_PROP = "player_prop"
    PARLAY = "parlay"
    TEASER = "teaser"


@dataclass
class BetRecord:
    """Record of a single bet."""

    bet_id: str
    placed_at: datetime
    game_id: str
    bet_type: BetType
    description: str  # e.g., "KC Chiefs -3.5"

    # Betting details
    bookmaker: str
    odds: int  # American odds
    line: Optional[float]  # Spread or total line
    stake: float

    # Model context
    model_probability: float
    edge: float
    expected_value: float

    # Status tracking
    status: BetStatus = BetStatus.PENDING
    settled_at: Optional[datetime] = None
    actual_outcome: Optional[float] = None  # Actual result (score diff, yards, etc.)
    payout: Optional[float] = None
    profit: Optional[float] = None

    # Metadata
    notes: str = ""
    tags: list[str] = field(default_factory=list)

    def settle(
        self,
        result: BetStatus,
        actual_outcome: Optional[float] = None,
    ) -> None:
        """Settle the bet with a result."""
        self.status = result
        self.settled_at = datetime.now()
        self.actual_outcome = actual_outcome

        if result == BetStatus.WON:
            # Calculate payout based on American odds
            if self.odds > 0:
                self.payout = self.stake * (1 + self.odds / 100)
            else:
                self.payout = self.stake * (1 + 100 / abs(self.odds))
            self.profit = self.payout - self.stake
        elif result == BetStatus.LOST:
            self.payout = 0.0
            self.profit = -self.stake
        elif result in (BetStatus.PUSH, BetStatus.VOIDED, BetStatus.CASHED_OUT):
            self.payout = self.stake
            self.profit = 0.0

    @property
    def is_settled(self) -> bool:
        return self.status != BetStatus.PENDING

    @property
    def potential_payout(self) -> float:
        """Calculate potential payout if bet wins."""
        if self.odds > 0:
            return self.stake * (1 + self.odds / 100)
        else:
            return self.stake * (1 + 100 / abs(self.odds))

    @property
    def potential_profit(self) -> float:
        """Calculate potential profit if bet wins."""
        return self.potential_payout - self.stake

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "bet_id": self.bet_id,
            "placed_at": self.placed_at.isoformat(),
            "game_id": self.game_id,
            "bet_type": self.bet_type.value,
            "description": self.description,
            "bookmaker": self.bookmaker,
            "odds": self.odds,
            "line": self.line,
            "stake": self.stake,
            "model_probability": self.model_probability,
            "edge": self.edge,
            "expected_value": self.expected_value,
            "status": self.status.value,
            "settled_at": self.settled_at.isoformat() if self.settled_at else None,
            "actual_outcome": self.actual_outcome,
            "payout": self.payout,
            "profit": self.profit,
            "notes": self.notes,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BetRecord":
        """Create from dictionary."""
        return cls(
            bet_id=data["bet_id"],
            placed_at=datetime.fromisoformat(data["placed_at"]),
            game_id=data["game_id"],
            bet_type=BetType(data["bet_type"]),
            description=data["description"],
            bookmaker=data["bookmaker"],
            odds=data["odds"],
            line=data.get("line"),
            stake=data["stake"],
            model_probability=data["model_probability"],
            edge=data["edge"],
            expected_value=data["expected_value"],
            status=BetStatus(data["status"]),
            settled_at=(
                datetime.fromisoformat(data["settled_at"])
                if data.get("settled_at")
                else None
            ),
            actual_outcome=data.get("actual_outcome"),
            payout=data.get("payout"),
            profit=data.get("profit"),
            notes=data.get("notes", ""),
            tags=data.get("tags", []),
        )


@dataclass
class PerformanceSummary:
    """Summary of betting performance."""

    total_bets: int
    pending_bets: int
    settled_bets: int

    # Win/Loss
    wins: int
    losses: int
    pushes: int
    win_rate: float

    # Financial
    total_staked: float
    total_returned: float
    net_profit: float
    roi: float  # Return on investment

    # Model performance
    avg_edge: float
    avg_expected_value: float
    actual_vs_expected: float  # How actual compares to expected

    # Risk metrics
    current_drawdown: float
    max_drawdown: float
    peak_bankroll: float

    # By bet type
    by_type: dict[str, dict] = field(default_factory=dict)


class BankrollManager:
    """
    Manages bankroll, tracks bets, and monitors risk.

    Provides comprehensive bankroll management including:
    - Balance tracking and pending exposure
    - Bet placement and settlement
    - Daily exposure limits
    - Drawdown monitoring
    - Performance analytics

    Example:
        >>> manager = BankrollManager(initial_bankroll=1000.0)
        >>> bet = manager.place_bet(value_bet, stake=25.0)
        >>> print(f"Available: ${manager.available_bankroll:.2f}")
        >>> manager.settle_bet(bet.bet_id, BetStatus.WON, actual_outcome=7)
        >>> print(f"New balance: ${manager.current_bankroll:.2f}")
    """

    def __init__(
        self,
        initial_bankroll: float = 1000.0,
        max_daily_risk: float = 0.20,
        max_single_bet: float = 0.05,
        data_dir: Optional[Path] = None,
    ):
        """
        Initialize the bankroll manager.

        Args:
            initial_bankroll: Starting bankroll in dollars
            max_daily_risk: Maximum daily exposure as % of bankroll
            max_single_bet: Maximum single bet as % of bankroll
            data_dir: Directory to persist bet history
        """
        self.initial_bankroll = initial_bankroll
        self.max_daily_risk = max_daily_risk
        self.max_single_bet = max_single_bet
        self.data_dir = data_dir

        # State
        self._bets: dict[str, BetRecord] = {}
        self._balance_history: list[tuple[datetime, float]] = [
            (datetime.now(), initial_bankroll)
        ]
        self._peak_bankroll = initial_bankroll

        # Load existing data if available
        if data_dir:
            self._load_data()

    @property
    def current_bankroll(self) -> float:
        """Current bankroll after all settled bets."""
        balance = self.initial_bankroll
        for bet in self._bets.values():
            if bet.is_settled and bet.profit is not None:
                balance += bet.profit
        return balance

    @property
    def pending_exposure(self) -> float:
        """Total amount at risk in pending bets."""
        return sum(
            bet.stake for bet in self._bets.values() if bet.status == BetStatus.PENDING
        )

    @property
    def available_bankroll(self) -> float:
        """Amount available for new bets."""
        return self.current_bankroll - self.pending_exposure

    @property
    def pending_bets(self) -> list[BetRecord]:
        """List of pending bets."""
        return [bet for bet in self._bets.values() if bet.status == BetStatus.PENDING]

    @property
    def settled_bets(self) -> list[BetRecord]:
        """List of settled bets."""
        return [bet for bet in self._bets.values() if bet.is_settled]

    def get_daily_exposure(self, date: Optional[datetime] = None) -> float:
        """
        Get total exposure for a specific day.

        Args:
            date: Date to check (defaults to today)

        Returns:
            Total stakes placed on that day
        """
        if date is None:
            date = datetime.now()

        start_of_day = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = start_of_day + timedelta(days=1)

        return sum(
            bet.stake
            for bet in self._bets.values()
            if start_of_day <= bet.placed_at < end_of_day
        )

    def get_remaining_daily_budget(self) -> float:
        """Get remaining amount that can be wagered today."""
        max_daily = self.max_daily_risk * self.current_bankroll
        used_today = self.get_daily_exposure()
        return max(0, max_daily - used_today)

    def can_place_bet(self, stake: float) -> tuple[bool, str]:
        """
        Check if a bet can be placed.

        Args:
            stake: Proposed stake amount

        Returns:
            Tuple of (can_place, reason_if_not)
        """
        if stake <= 0:
            return False, "Stake must be positive"

        if stake > self.available_bankroll:
            return (
                False,
                f"Insufficient funds. Available: ${self.available_bankroll:.2f}",
            )

        max_stake = self.max_single_bet * self.current_bankroll
        if stake > max_stake:
            return (
                False,
                f"Exceeds max stake ({self.max_single_bet:.0%}). Max: ${max_stake:.2f}",
            )

        remaining_daily = self.get_remaining_daily_budget()
        if stake > remaining_daily:
            return (
                False,
                f"Exceeds daily limit. Remaining: ${remaining_daily:.2f}",
            )

        return True, ""

    def place_bet(
        self,
        game_id: str,
        bet_type: BetType,
        description: str,
        bookmaker: str,
        odds: int,
        stake: float,
        model_probability: float,
        edge: float,
        expected_value: float,
        line: Optional[float] = None,
        notes: str = "",
        tags: Optional[list[str]] = None,
    ) -> BetRecord:
        """
        Record a placed bet.

        Args:
            game_id: Game identifier
            bet_type: Type of bet
            description: Human-readable description
            bookmaker: Sportsbook name
            odds: American odds
            stake: Bet amount in dollars
            model_probability: Model's win probability
            edge: Calculated edge
            expected_value: Expected value per dollar
            line: Spread or total line
            notes: Optional notes
            tags: Optional tags for categorization

        Returns:
            BetRecord for the placed bet

        Raises:
            ValueError: If bet cannot be placed
        """
        can_place, reason = self.can_place_bet(stake)
        if not can_place:
            raise ValueError(f"Cannot place bet: {reason}")

        bet_id = f"bet_{int(datetime.now().timestamp() * 1000)}"

        bet = BetRecord(
            bet_id=bet_id,
            placed_at=datetime.now(),
            game_id=game_id,
            bet_type=bet_type,
            description=description,
            bookmaker=bookmaker,
            odds=odds,
            line=line,
            stake=stake,
            model_probability=model_probability,
            edge=edge,
            expected_value=expected_value,
            notes=notes,
            tags=tags or [],
        )

        self._bets[bet_id] = bet
        self._save_data()

        return bet

    def settle_bet(
        self,
        bet_id: str,
        result: BetStatus,
        actual_outcome: Optional[float] = None,
    ) -> BetRecord:
        """
        Settle a pending bet.

        Args:
            bet_id: ID of the bet to settle
            result: Result of the bet
            actual_outcome: Actual game outcome for analysis

        Returns:
            Updated BetRecord

        Raises:
            ValueError: If bet not found or already settled
        """
        if bet_id not in self._bets:
            raise ValueError(f"Bet not found: {bet_id}")

        bet = self._bets[bet_id]
        if bet.is_settled:
            raise ValueError(f"Bet already settled: {bet_id}")

        bet.settle(result, actual_outcome)

        # Update peak bankroll
        current = self.current_bankroll
        if current > self._peak_bankroll:
            self._peak_bankroll = current

        # Record balance history
        self._balance_history.append((datetime.now(), current))

        self._save_data()

        return bet

    def get_bet(self, bet_id: str) -> Optional[BetRecord]:
        """Get a specific bet by ID."""
        return self._bets.get(bet_id)

    def get_drawdown(self) -> float:
        """
        Calculate current drawdown from peak.

        Returns:
            Drawdown as a decimal (0.15 = 15% drawdown)
        """
        current = self.current_bankroll
        if self._peak_bankroll <= 0:
            return 0.0
        return (self._peak_bankroll - current) / self._peak_bankroll

    def get_max_drawdown(self) -> float:
        """
        Calculate maximum historical drawdown.

        Returns:
            Max drawdown as a decimal
        """
        if len(self._balance_history) < 2:
            return 0.0

        peak = self.initial_bankroll
        max_dd = 0.0

        for _, balance in self._balance_history:
            if balance > peak:
                peak = balance
            dd = (peak - balance) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

        return max_dd

    def get_performance_summary(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> PerformanceSummary:
        """
        Get comprehensive performance summary.

        Args:
            start_date: Start of period (defaults to all time)
            end_date: End of period (defaults to now)

        Returns:
            PerformanceSummary with all metrics
        """
        # Filter bets by date range
        bets = list(self._bets.values())
        if start_date:
            bets = [b for b in bets if b.placed_at >= start_date]
        if end_date:
            bets = [b for b in bets if b.placed_at <= end_date]

        settled = [b for b in bets if b.is_settled]
        pending = [b for b in bets if not b.is_settled]

        wins = [b for b in settled if b.status == BetStatus.WON]
        losses = [b for b in settled if b.status == BetStatus.LOST]
        pushes = [b for b in settled if b.status == BetStatus.PUSH]

        total_staked = sum(b.stake for b in settled)
        total_returned = sum(b.payout or 0 for b in settled)
        net_profit = sum(b.profit or 0 for b in settled)

        win_rate = len(wins) / len(settled) if settled else 0.0
        roi = net_profit / total_staked if total_staked > 0 else 0.0

        avg_edge = sum(b.edge for b in bets) / len(bets) if bets else 0.0
        avg_ev = (
            sum(b.expected_value for b in bets) / len(bets) if bets else 0.0
        )

        # Expected profit based on model probabilities
        expected_profit = sum(b.expected_value * b.stake for b in settled)
        actual_vs_expected = (
            (net_profit - expected_profit) / abs(expected_profit)
            if expected_profit != 0
            else 0.0
        )

        # By bet type
        by_type = {}
        for bet_type in BetType:
            type_bets = [b for b in settled if b.bet_type == bet_type]
            if type_bets:
                type_wins = sum(1 for b in type_bets if b.status == BetStatus.WON)
                type_profit = sum(b.profit or 0 for b in type_bets)
                type_staked = sum(b.stake for b in type_bets)
                by_type[bet_type.value] = {
                    "count": len(type_bets),
                    "wins": type_wins,
                    "win_rate": type_wins / len(type_bets),
                    "profit": type_profit,
                    "roi": type_profit / type_staked if type_staked > 0 else 0,
                }

        return PerformanceSummary(
            total_bets=len(bets),
            pending_bets=len(pending),
            settled_bets=len(settled),
            wins=len(wins),
            losses=len(losses),
            pushes=len(pushes),
            win_rate=win_rate,
            total_staked=total_staked,
            total_returned=total_returned,
            net_profit=net_profit,
            roi=roi,
            avg_edge=avg_edge,
            avg_expected_value=avg_ev,
            actual_vs_expected=actual_vs_expected,
            current_drawdown=self.get_drawdown(),
            max_drawdown=self.get_max_drawdown(),
            peak_bankroll=self._peak_bankroll,
            by_type=by_type,
        )

    def get_bets_by_game(self, game_id: str) -> list[BetRecord]:
        """Get all bets for a specific game."""
        return [b for b in self._bets.values() if b.game_id == game_id]

    def get_bets_by_status(self, status: BetStatus) -> list[BetRecord]:
        """Get all bets with a specific status."""
        return [b for b in self._bets.values() if b.status == status]

    def get_recent_bets(self, days: int = 7) -> list[BetRecord]:
        """Get bets from the last N days."""
        cutoff = datetime.now() - timedelta(days=days)
        return [b for b in self._bets.values() if b.placed_at >= cutoff]

    def _save_data(self) -> None:
        """Save bet history to disk."""
        if not self.data_dir:
            return

        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Save bets
        bets_file = self.data_dir / "bets.json"
        bets_data = [bet.to_dict() for bet in self._bets.values()]
        with open(bets_file, "w") as f:
            json.dump(bets_data, f, indent=2)

        # Save balance history
        history_file = self.data_dir / "balance_history.json"
        history_data = [
            {"timestamp": ts.isoformat(), "balance": balance}
            for ts, balance in self._balance_history
        ]
        with open(history_file, "w") as f:
            json.dump(history_data, f, indent=2)

        # Save metadata
        meta_file = self.data_dir / "metadata.json"
        meta_data = {
            "initial_bankroll": self.initial_bankroll,
            "peak_bankroll": self._peak_bankroll,
            "max_daily_risk": self.max_daily_risk,
            "max_single_bet": self.max_single_bet,
        }
        with open(meta_file, "w") as f:
            json.dump(meta_data, f, indent=2)

    def _load_data(self) -> None:
        """Load bet history from disk."""
        if not self.data_dir or not self.data_dir.exists():
            return

        # Load bets
        bets_file = self.data_dir / "bets.json"
        if bets_file.exists():
            with open(bets_file) as f:
                bets_data = json.load(f)
            self._bets = {
                bet_data["bet_id"]: BetRecord.from_dict(bet_data)
                for bet_data in bets_data
            }

        # Load balance history
        history_file = self.data_dir / "balance_history.json"
        if history_file.exists():
            with open(history_file) as f:
                history_data = json.load(f)
            self._balance_history = [
                (datetime.fromisoformat(item["timestamp"]), item["balance"])
                for item in history_data
            ]

        # Load metadata
        meta_file = self.data_dir / "metadata.json"
        if meta_file.exists():
            with open(meta_file) as f:
                meta_data = json.load(f)
            self._peak_bankroll = meta_data.get("peak_bankroll", self.initial_bankroll)

    def export_to_csv(self, filepath: Path) -> None:
        """Export bet history to CSV."""
        import csv

        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "bet_id",
                    "placed_at",
                    "game_id",
                    "bet_type",
                    "description",
                    "bookmaker",
                    "odds",
                    "line",
                    "stake",
                    "model_probability",
                    "edge",
                    "expected_value",
                    "status",
                    "settled_at",
                    "actual_outcome",
                    "payout",
                    "profit",
                ]
            )
            for bet in self._bets.values():
                writer.writerow(
                    [
                        bet.bet_id,
                        bet.placed_at.isoformat(),
                        bet.game_id,
                        bet.bet_type.value,
                        bet.description,
                        bet.bookmaker,
                        bet.odds,
                        bet.line,
                        bet.stake,
                        bet.model_probability,
                        bet.edge,
                        bet.expected_value,
                        bet.status.value,
                        bet.settled_at.isoformat() if bet.settled_at else "",
                        bet.actual_outcome,
                        bet.payout,
                        bet.profit,
                    ]
                )
