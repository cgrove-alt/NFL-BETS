"""
Cross-book arbitrage scanner.

Identifies guaranteed profit opportunities by finding
complementary odds across different bookmakers.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

from .odds_converter import (
    american_to_decimal,
    american_to_implied_probability,
    calculate_arbitrage_stakes,
    check_arbitrage,
)


class ArbType(str, Enum):
    """Types of arbitrage opportunities."""

    SPREAD = "spread"
    MONEYLINE = "moneyline"
    TOTAL = "total"
    PLAYER_PROP = "player_prop"


@dataclass
class ArbitrageOpportunity:
    """Represents a guaranteed profit arbitrage opportunity."""

    arb_id: str
    arb_type: ArbType
    game_id: str
    description: str  # e.g., "KC Chiefs vs BAL Ravens - Spread"

    # Leg 1 (e.g., Home team or Over)
    book1: str
    side1: str  # e.g., "KC -3.5" or "Over 45.5"
    odds1: int  # American odds

    # Leg 2 (e.g., Away team or Under)
    book2: str
    side2: str  # e.g., "BAL +3.5" or "Under 45.5"
    odds2: int  # American odds

    # Profit metrics
    arb_percentage: float  # Sum of implied probs (< 100% = arb exists)
    profit_percentage: float  # Guaranteed profit as % of total stake

    # Optimal stakes for $100 total
    stake1: float
    stake2: float
    total_stake: float
    guaranteed_profit: float

    # Timing
    detected_at: datetime = field(default_factory=datetime.now)
    game_time: Optional[datetime] = None
    expires_at: Optional[datetime] = None

    # Metadata
    line: Optional[float] = None  # Spread or total line
    is_same_line: bool = True  # Both books have same line

    def scale_stakes(self, target_total: float) -> tuple[float, float, float]:
        """
        Scale stakes to a different total amount.

        Args:
            target_total: Desired total stake

        Returns:
            Tuple of (stake1, stake2, guaranteed_profit)
        """
        scale = target_total / self.total_stake
        return (
            round(self.stake1 * scale, 2),
            round(self.stake2 * scale, 2),
            round(self.guaranteed_profit * scale, 2),
        )

    @property
    def roi(self) -> float:
        """Return on investment as a decimal."""
        if self.total_stake == 0:
            return 0.0
        return self.guaranteed_profit / self.total_stake

    @property
    def time_to_game(self) -> Optional[float]:
        """Hours until game starts."""
        if self.game_time is None:
            return None
        delta = self.game_time - datetime.now()
        return delta.total_seconds() / 3600


@dataclass
class BestLine:
    """Best available line from any bookmaker."""

    side: str  # e.g., "KC -3.5"
    bookmaker: str
    odds: int
    line: Optional[float] = None


@dataclass
class ScanResult:
    """Results from scanning for arbitrage opportunities."""

    opportunities: list[ArbitrageOpportunity] = field(default_factory=list)
    best_lines: dict[str, dict[str, BestLine]] = field(
        default_factory=dict
    )  # game_id -> side -> best_line
    scanned_games: int = 0
    scanned_books: int = 0
    scan_time: datetime = field(default_factory=datetime.now)

    @property
    def has_opportunities(self) -> bool:
        return len(self.opportunities) > 0

    def get_top_opportunities(self, n: int = 5) -> list[ArbitrageOpportunity]:
        """Get top N opportunities sorted by profit percentage."""
        return sorted(
            self.opportunities, key=lambda x: x.profit_percentage, reverse=True
        )[:n]


class ArbitrageScanner:
    """
    Scanner for cross-book arbitrage opportunities.

    Arbitrage exists when the sum of implied probabilities
    across bookmakers is less than 100%. This guarantees
    a profit regardless of outcome.

    Example:
        Book A: KC -3.5 @ +105 (implied 48.8%)
        Book B: BAL +3.5 @ +105 (implied 48.8%)
        Total implied: 97.6% < 100% = 2.4% guaranteed profit

    Usage:
        >>> scanner = ArbitrageScanner(min_profit_pct=0.005)
        >>> result = scanner.scan_spreads(odds_by_book)
        >>> for arb in result.opportunities:
        ...     print(f"{arb.profit_percentage:.1%} profit on {arb.description}")
    """

    def __init__(
        self,
        min_profit_pct: float = 0.005,  # 0.5% minimum profit
        max_total_stake: float = 1000.0,
        include_same_book: bool = False,  # Usually can't arb same book
    ):
        """
        Initialize the arbitrage scanner.

        Args:
            min_profit_pct: Minimum profit percentage to report (0.005 = 0.5%)
            max_total_stake: Default maximum total stake for calculations
            include_same_book: Whether to look for arbs within same bookmaker
        """
        self.min_profit_pct = min_profit_pct
        self.max_total_stake = max_total_stake
        self.include_same_book = include_same_book

    def check_arbitrage_opportunity(
        self,
        odds1: int,
        odds2: int,
    ) -> tuple[bool, float, float, float]:
        """
        Check if arbitrage exists between two odds.

        Args:
            odds1: American odds for side 1
            odds2: American odds for side 2

        Returns:
            Tuple of (exists, profit_pct, stake1_pct, stake2_pct)
        """
        exists, profit_pct = check_arbitrage(odds1, odds2)

        if not exists or profit_pct < self.min_profit_pct:
            return (False, 0.0, 0.0, 0.0)

        stake1, stake2 = calculate_arbitrage_stakes(odds1, odds2, 100.0)
        stake1_pct = stake1 / 100.0
        stake2_pct = stake2 / 100.0

        return (True, profit_pct, stake1_pct, stake2_pct)

    def scan_spreads(
        self,
        odds_by_book: dict[str, list[dict]],
    ) -> ScanResult:
        """
        Scan spread markets for arbitrage opportunities.

        Args:
            odds_by_book: Dict mapping bookmaker name to list of game odds.
                Each game odds dict should have:
                - game_id: str
                - home_team: str
                - away_team: str
                - home_spread: float
                - home_spread_odds: int
                - away_spread_odds: int
                - game_time: datetime (optional)

        Returns:
            ScanResult with found opportunities and best lines
        """
        opportunities: list[ArbitrageOpportunity] = []
        best_lines: dict[str, dict[str, BestLine]] = {}

        # Build odds lookup by game_id and side
        games_by_id: dict[str, dict[str, list[tuple[str, int, float]]]] = {}

        for book, games in odds_by_book.items():
            for game in games:
                game_id = game["game_id"]
                if game_id not in games_by_id:
                    games_by_id[game_id] = {
                        "home": [],
                        "away": [],
                        "meta": {
                            "home_team": game.get("home_team", ""),
                            "away_team": game.get("away_team", ""),
                            "game_time": game.get("game_time"),
                        },
                    }

                home_spread = game.get("home_spread", 0)
                home_odds = game.get("home_spread_odds")
                away_odds = game.get("away_spread_odds")

                if home_odds is not None:
                    games_by_id[game_id]["home"].append((book, home_odds, home_spread))
                if away_odds is not None:
                    away_spread = -home_spread
                    games_by_id[game_id]["away"].append((book, away_odds, away_spread))

        # Find best lines and arbitrage for each game
        for game_id, sides in games_by_id.items():
            if "meta" not in sides:
                continue

            meta = sides["meta"]
            home_team = meta.get("home_team", "Home")
            away_team = meta.get("away_team", "Away")
            game_time = meta.get("game_time")

            best_lines[game_id] = {}

            # Find best home spread (highest odds)
            home_bets = sides.get("home", [])
            if home_bets:
                best_home = max(home_bets, key=lambda x: x[1])
                best_lines[game_id]["home"] = BestLine(
                    side=f"{home_team} {best_home[2]:+.1f}",
                    bookmaker=best_home[0],
                    odds=best_home[1],
                    line=best_home[2],
                )

            # Find best away spread (highest odds)
            away_bets = sides.get("away", [])
            if away_bets:
                best_away = max(away_bets, key=lambda x: x[1])
                best_lines[game_id]["away"] = BestLine(
                    side=f"{away_team} {best_away[2]:+.1f}",
                    bookmaker=best_away[0],
                    odds=best_away[1],
                    line=best_away[2],
                )

            # Check for arbitrage between best lines
            if "home" in best_lines[game_id] and "away" in best_lines[game_id]:
                best_home_line = best_lines[game_id]["home"]
                best_away_line = best_lines[game_id]["away"]

                # Skip if same book and not allowed
                if (
                    not self.include_same_book
                    and best_home_line.bookmaker == best_away_line.bookmaker
                ):
                    continue

                exists, profit_pct, stake1_pct, stake2_pct = (
                    self.check_arbitrage_opportunity(
                        best_home_line.odds, best_away_line.odds
                    )
                )

                if exists:
                    total_stake = 100.0
                    stake1 = round(stake1_pct * total_stake, 2)
                    stake2 = round(stake2_pct * total_stake, 2)
                    profit = round(profit_pct * total_stake, 2)

                    # Calculate arb percentage
                    impl1 = american_to_implied_probability(best_home_line.odds)
                    impl2 = american_to_implied_probability(best_away_line.odds)
                    arb_pct = impl1 + impl2

                    arb = ArbitrageOpportunity(
                        arb_id=f"arb_spread_{game_id}_{int(datetime.now().timestamp())}",
                        arb_type=ArbType.SPREAD,
                        game_id=game_id,
                        description=f"{home_team} vs {away_team} - Spread",
                        book1=best_home_line.bookmaker,
                        side1=best_home_line.side,
                        odds1=best_home_line.odds,
                        book2=best_away_line.bookmaker,
                        side2=best_away_line.side,
                        odds2=best_away_line.odds,
                        arb_percentage=arb_pct,
                        profit_percentage=profit_pct,
                        stake1=stake1,
                        stake2=stake2,
                        total_stake=total_stake,
                        guaranteed_profit=profit,
                        game_time=game_time,
                        line=best_home_line.line,
                        is_same_line=abs(best_home_line.line + best_away_line.line)
                        < 0.01,
                    )
                    opportunities.append(arb)

        return ScanResult(
            opportunities=opportunities,
            best_lines=best_lines,
            scanned_games=len(games_by_id),
            scanned_books=len(odds_by_book),
        )

    def scan_totals(
        self,
        odds_by_book: dict[str, list[dict]],
    ) -> ScanResult:
        """
        Scan totals (over/under) markets for arbitrage.

        Args:
            odds_by_book: Dict mapping bookmaker to list of game odds.
                Each game odds dict should have:
                - game_id: str
                - home_team: str
                - away_team: str
                - total: float
                - over_odds: int
                - under_odds: int
                - game_time: datetime (optional)

        Returns:
            ScanResult with opportunities
        """
        opportunities: list[ArbitrageOpportunity] = []
        best_lines: dict[str, dict[str, BestLine]] = {}

        # Build lookup by game
        games_by_id: dict[str, dict[str, list[tuple[str, int, float]]]] = {}

        for book, games in odds_by_book.items():
            for game in games:
                game_id = game["game_id"]
                if game_id not in games_by_id:
                    games_by_id[game_id] = {
                        "over": [],
                        "under": [],
                        "meta": {
                            "home_team": game.get("home_team", ""),
                            "away_team": game.get("away_team", ""),
                            "game_time": game.get("game_time"),
                        },
                    }

                total = game.get("total", 0)
                over_odds = game.get("over_odds")
                under_odds = game.get("under_odds")

                if over_odds is not None:
                    games_by_id[game_id]["over"].append((book, over_odds, total))
                if under_odds is not None:
                    games_by_id[game_id]["under"].append((book, under_odds, total))

        # Find arbs
        for game_id, sides in games_by_id.items():
            if "meta" not in sides:
                continue

            meta = sides["meta"]
            home_team = meta.get("home_team", "Home")
            away_team = meta.get("away_team", "Away")
            game_time = meta.get("game_time")

            best_lines[game_id] = {}

            over_bets = sides.get("over", [])
            if over_bets:
                best_over = max(over_bets, key=lambda x: x[1])
                best_lines[game_id]["over"] = BestLine(
                    side=f"Over {best_over[2]}",
                    bookmaker=best_over[0],
                    odds=best_over[1],
                    line=best_over[2],
                )

            under_bets = sides.get("under", [])
            if under_bets:
                best_under = max(under_bets, key=lambda x: x[1])
                best_lines[game_id]["under"] = BestLine(
                    side=f"Under {best_under[2]}",
                    bookmaker=best_under[0],
                    odds=best_under[1],
                    line=best_under[2],
                )

            if "over" in best_lines[game_id] and "under" in best_lines[game_id]:
                best_over_line = best_lines[game_id]["over"]
                best_under_line = best_lines[game_id]["under"]

                if (
                    not self.include_same_book
                    and best_over_line.bookmaker == best_under_line.bookmaker
                ):
                    continue

                exists, profit_pct, stake1_pct, stake2_pct = (
                    self.check_arbitrage_opportunity(
                        best_over_line.odds, best_under_line.odds
                    )
                )

                if exists:
                    total_stake = 100.0
                    stake1 = round(stake1_pct * total_stake, 2)
                    stake2 = round(stake2_pct * total_stake, 2)
                    profit = round(profit_pct * total_stake, 2)

                    impl1 = american_to_implied_probability(best_over_line.odds)
                    impl2 = american_to_implied_probability(best_under_line.odds)

                    arb = ArbitrageOpportunity(
                        arb_id=f"arb_total_{game_id}_{int(datetime.now().timestamp())}",
                        arb_type=ArbType.TOTAL,
                        game_id=game_id,
                        description=f"{home_team} vs {away_team} - Total",
                        book1=best_over_line.bookmaker,
                        side1=best_over_line.side,
                        odds1=best_over_line.odds,
                        book2=best_under_line.bookmaker,
                        side2=best_under_line.side,
                        odds2=best_under_line.odds,
                        arb_percentage=impl1 + impl2,
                        profit_percentage=profit_pct,
                        stake1=stake1,
                        stake2=stake2,
                        total_stake=total_stake,
                        guaranteed_profit=profit,
                        game_time=game_time,
                        line=best_over_line.line,
                        is_same_line=abs(best_over_line.line - best_under_line.line)
                        < 0.01,
                    )
                    opportunities.append(arb)

        return ScanResult(
            opportunities=opportunities,
            best_lines=best_lines,
            scanned_games=len(games_by_id),
            scanned_books=len(odds_by_book),
        )

    def scan_moneylines(
        self,
        odds_by_book: dict[str, list[dict]],
    ) -> ScanResult:
        """
        Scan moneyline markets for arbitrage.

        Args:
            odds_by_book: Dict mapping bookmaker to list of game odds.
                Each game odds dict should have:
                - game_id: str
                - home_team: str
                - away_team: str
                - home_ml: int
                - away_ml: int
                - game_time: datetime (optional)

        Returns:
            ScanResult with opportunities
        """
        opportunities: list[ArbitrageOpportunity] = []
        best_lines: dict[str, dict[str, BestLine]] = {}

        games_by_id: dict[str, dict[str, list[tuple[str, int]]]] = {}

        for book, games in odds_by_book.items():
            for game in games:
                game_id = game["game_id"]
                if game_id not in games_by_id:
                    games_by_id[game_id] = {
                        "home": [],
                        "away": [],
                        "meta": {
                            "home_team": game.get("home_team", ""),
                            "away_team": game.get("away_team", ""),
                            "game_time": game.get("game_time"),
                        },
                    }

                home_ml = game.get("home_ml")
                away_ml = game.get("away_ml")

                if home_ml is not None:
                    games_by_id[game_id]["home"].append((book, home_ml))
                if away_ml is not None:
                    games_by_id[game_id]["away"].append((book, away_ml))

        for game_id, sides in games_by_id.items():
            if "meta" not in sides:
                continue

            meta = sides["meta"]
            home_team = meta.get("home_team", "Home")
            away_team = meta.get("away_team", "Away")
            game_time = meta.get("game_time")

            best_lines[game_id] = {}

            home_bets = sides.get("home", [])
            if home_bets:
                best_home = max(home_bets, key=lambda x: x[1])
                best_lines[game_id]["home"] = BestLine(
                    side=f"{home_team} ML",
                    bookmaker=best_home[0],
                    odds=best_home[1],
                )

            away_bets = sides.get("away", [])
            if away_bets:
                best_away = max(away_bets, key=lambda x: x[1])
                best_lines[game_id]["away"] = BestLine(
                    side=f"{away_team} ML",
                    bookmaker=best_away[0],
                    odds=best_away[1],
                )

            if "home" in best_lines[game_id] and "away" in best_lines[game_id]:
                best_home_line = best_lines[game_id]["home"]
                best_away_line = best_lines[game_id]["away"]

                if (
                    not self.include_same_book
                    and best_home_line.bookmaker == best_away_line.bookmaker
                ):
                    continue

                exists, profit_pct, stake1_pct, stake2_pct = (
                    self.check_arbitrage_opportunity(
                        best_home_line.odds, best_away_line.odds
                    )
                )

                if exists:
                    total_stake = 100.0
                    stake1 = round(stake1_pct * total_stake, 2)
                    stake2 = round(stake2_pct * total_stake, 2)
                    profit = round(profit_pct * total_stake, 2)

                    impl1 = american_to_implied_probability(best_home_line.odds)
                    impl2 = american_to_implied_probability(best_away_line.odds)

                    arb = ArbitrageOpportunity(
                        arb_id=f"arb_ml_{game_id}_{int(datetime.now().timestamp())}",
                        arb_type=ArbType.MONEYLINE,
                        game_id=game_id,
                        description=f"{home_team} vs {away_team} - Moneyline",
                        book1=best_home_line.bookmaker,
                        side1=best_home_line.side,
                        odds1=best_home_line.odds,
                        book2=best_away_line.bookmaker,
                        side2=best_away_line.side,
                        odds2=best_away_line.odds,
                        arb_percentage=impl1 + impl2,
                        profit_percentage=profit_pct,
                        stake1=stake1,
                        stake2=stake2,
                        total_stake=total_stake,
                        guaranteed_profit=profit,
                        game_time=game_time,
                    )
                    opportunities.append(arb)

        return ScanResult(
            opportunities=opportunities,
            best_lines=best_lines,
            scanned_games=len(games_by_id),
            scanned_books=len(odds_by_book),
        )

    def scan_all_markets(
        self,
        odds_by_book: dict[str, list[dict]],
    ) -> ScanResult:
        """
        Scan all markets (spreads, totals, moneylines) for arbitrage.

        Args:
            odds_by_book: Dict with all market types included

        Returns:
            Combined ScanResult with all opportunities
        """
        spread_result = self.scan_spreads(odds_by_book)
        total_result = self.scan_totals(odds_by_book)
        ml_result = self.scan_moneylines(odds_by_book)

        all_opportunities = (
            spread_result.opportunities
            + total_result.opportunities
            + ml_result.opportunities
        )

        # Merge best lines
        all_best_lines = {}
        for result in [spread_result, total_result, ml_result]:
            for game_id, lines in result.best_lines.items():
                if game_id not in all_best_lines:
                    all_best_lines[game_id] = {}
                all_best_lines[game_id].update(lines)

        return ScanResult(
            opportunities=sorted(
                all_opportunities, key=lambda x: x.profit_percentage, reverse=True
            ),
            best_lines=all_best_lines,
            scanned_games=spread_result.scanned_games,
            scanned_books=spread_result.scanned_books,
        )

    def find_best_lines(
        self,
        odds_by_book: dict[str, list[dict]],
        game_id: str,
    ) -> dict[str, BestLine]:
        """
        Find best available lines for a specific game.

        Args:
            odds_by_book: Dict mapping bookmaker to games
            game_id: Game ID to find lines for

        Returns:
            Dict mapping side (home, away, over, under) to best line
        """
        result = self.scan_all_markets(odds_by_book)
        return result.best_lines.get(game_id, {})
