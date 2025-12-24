"""
Value bet detection for NFL betting.

Identifies +EV betting opportunities by comparing model predictions
to market odds. Core component that translates ML predictions into
actionable betting recommendations.
"""
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

from .odds_converter import (
    american_to_decimal,
    american_to_implied_probability,
    calculate_edge,
    calculate_expected_value,
    remove_vig,
)


class Urgency(str, Enum):
    """Urgency levels for value bet alerts."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class BetType(str, Enum):
    """Types of bets."""

    SPREAD = "spread"
    MONEYLINE = "moneyline"
    TOTAL = "total"
    PASSING_YARDS = "passing_yards"
    RUSHING_YARDS = "rushing_yards"
    RECEIVING_YARDS = "receiving_yards"
    RECEPTIONS = "receptions"
    TOUCHDOWNS = "touchdowns"


@dataclass
class ValueBet:
    """
    Container for a detected value betting opportunity.

    Includes all information needed to evaluate and place the bet.
    """

    # Identification
    bet_id: str
    bet_type: BetType
    game_id: str
    description: str  # Human-readable: "KC Chiefs -3.5"

    # Model prediction
    model_probability: float
    model_prediction: float  # Predicted spread or yards

    # Market information
    bookmaker: str
    odds: int  # American odds (-110)
    implied_probability: float
    line: float

    # Edge metrics
    edge: float  # model_prob - no_vig_implied_prob
    expected_value: float  # EV per dollar wagered
    raw_edge: float  # edge vs raw implied (with vig)

    # Recommendation
    recommended_stake: float = 0.0  # Filled by Kelly calculator
    stake_percentage: float = 0.0  # As % of bankroll
    urgency: Urgency = Urgency.MEDIUM

    # Context
    home_team: Optional[str] = None
    away_team: Optional[str] = None
    player_name: Optional[str] = None
    player_id: Optional[str] = None

    # Timing
    detected_at: datetime = field(default_factory=datetime.now)
    game_time: Optional[datetime] = None

    @property
    def time_until_game(self) -> Optional[timedelta]:
        """Time remaining until game starts."""
        if self.game_time is None:
            return None
        return self.game_time - datetime.now()

    @property
    def is_value(self) -> bool:
        """Check if this is a valid value bet (edge > 0)."""
        return self.edge > 0

    @property
    def pick_description(self) -> str:
        """Get formatted pick description."""
        odds_str = f"+{self.odds}" if self.odds > 0 else str(self.odds)
        return f"{self.description} @ {odds_str} ({self.bookmaker})"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "bet_id": self.bet_id,
            "bet_type": self.bet_type.value,
            "game_id": self.game_id,
            "description": self.description,
            "model_probability": self.model_probability,
            "model_prediction": self.model_prediction,
            "bookmaker": self.bookmaker,
            "odds": self.odds,
            "implied_probability": self.implied_probability,
            "line": self.line,
            "edge": self.edge,
            "expected_value": self.expected_value,
            "recommended_stake": self.recommended_stake,
            "stake_percentage": self.stake_percentage,
            "urgency": self.urgency.value,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "player_name": self.player_name,
            "detected_at": self.detected_at.isoformat(),
            "game_time": self.game_time.isoformat() if self.game_time else None,
        }

    def summary(self) -> str:
        """Get formatted summary string."""
        lines = [
            f"[{self.urgency.value}] {self.pick_description}",
            f"  Edge: {self.edge:.1%} | EV: {self.expected_value:+.1%}",
            f"  Model: {self.model_probability:.1%} vs Market: {self.implied_probability:.1%}",
        ]
        if self.recommended_stake > 0:
            lines.append(f"  Stake: ${self.recommended_stake:.2f} ({self.stake_percentage:.1%})")
        return "\n".join(lines)


@dataclass
class DetectionResult:
    """Result of a value detection scan."""

    value_bets: list[ValueBet]
    scanned_at: datetime = field(default_factory=datetime.now)
    games_scanned: int = 0
    props_scanned: int = 0
    total_opportunities: int = 0

    @property
    def high_priority_bets(self) -> list[ValueBet]:
        """Get HIGH and CRITICAL urgency bets."""
        return [
            b for b in self.value_bets
            if b.urgency in (Urgency.HIGH, Urgency.CRITICAL)
        ]

    @property
    def best_edge(self) -> Optional[ValueBet]:
        """Get bet with highest edge."""
        if not self.value_bets:
            return None
        return max(self.value_bets, key=lambda b: b.edge)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "scanned_at": self.scanned_at.isoformat(),
            "games_scanned": self.games_scanned,
            "props_scanned": self.props_scanned,
            "total_opportunities": self.total_opportunities,
            "value_bets": [b.to_dict() for b in self.value_bets],
        }


class ValueDetector:
    """
    Detects value betting opportunities by comparing model predictions
    to market odds.

    Example:
        >>> detector = ValueDetector(
        ...     spread_model=spread_model,
        ...     min_edge=0.03,
        ...     min_ev=0.02,
        ... )
        >>> result = detector.scan_spreads(games, odds, features)
        >>> for bet in result.value_bets:
        ...     print(bet.summary())
    """

    # Default thresholds
    DEFAULT_MIN_EDGE = 0.03  # 3% minimum edge
    DEFAULT_MIN_EV = 0.02  # 2% minimum expected value

    # Urgency thresholds
    CRITICAL_EDGE = 0.08  # 8%
    HIGH_EDGE = 0.05  # 5%
    CRITICAL_HOURS = 2
    HIGH_HOURS = 6

    def __init__(
        self,
        spread_model=None,
        prop_models: Optional[dict] = None,
        min_edge: float = DEFAULT_MIN_EDGE,
        min_ev: float = DEFAULT_MIN_EV,
        remove_vig_in_comparison: bool = True,
    ):
        """
        Initialize the value detector.

        Args:
            spread_model: Trained SpreadModel for game spread predictions
            prop_models: Dict mapping prop_type to trained prop models
            min_edge: Minimum edge required to flag as value (default 3%)
            min_ev: Minimum expected value required (default 2%)
            remove_vig_in_comparison: If True, compare to no-vig probabilities
        """
        self.spread_model = spread_model
        self.prop_models = prop_models or {}
        self.min_edge = min_edge
        self.min_ev = min_ev
        self.remove_vig = remove_vig_in_comparison

    def scan_spreads(
        self,
        games: list[dict],
        odds_data: list[dict],
        features: dict[str, dict],
    ) -> DetectionResult:
        """
        Scan spread bets for value opportunities.

        Args:
            games: List of game dicts with game_id, home_team, away_team, game_time
            odds_data: List of odds dicts with game_id, bookmaker, home_spread, home_odds, away_odds
            features: Dict mapping game_id to feature dict

        Returns:
            DetectionResult with detected value bets
        """
        if self.spread_model is None:
            logger.warning("No spread model configured")
            return DetectionResult(value_bets=[], games_scanned=0)

        value_bets = []

        for game in games:
            game_id = game.get("game_id")
            if not game_id:
                continue

            game_features = features.get(game_id)
            if game_features is None:
                logger.debug(f"No features for game {game_id}")
                continue

            # Get model prediction
            try:
                prediction = self.spread_model.predict_game(
                    features=game_features,
                    game_id=game_id,
                    home_team=game.get("home_team", ""),
                    away_team=game.get("away_team", ""),
                )
            except Exception as e:
                logger.warning(f"Failed to predict {game_id}: {e}")
                continue

            # Find odds for this game
            game_odds = [o for o in odds_data if o.get("game_id") == game_id]

            for odds in game_odds:
                # Check home spread
                home_bet = self._evaluate_spread_bet(
                    prediction=prediction,
                    game=game,
                    odds=odds,
                    side="home",
                )
                if home_bet:
                    value_bets.append(home_bet)

                # Check away spread
                away_bet = self._evaluate_spread_bet(
                    prediction=prediction,
                    game=game,
                    odds=odds,
                    side="away",
                )
                if away_bet:
                    value_bets.append(away_bet)

        # Sort by edge
        value_bets.sort(key=lambda b: b.edge, reverse=True)

        return DetectionResult(
            value_bets=value_bets,
            games_scanned=len(games),
            total_opportunities=len(value_bets),
        )

    def _evaluate_spread_bet(
        self,
        prediction,
        game: dict,
        odds: dict,
        side: str,
    ) -> Optional[ValueBet]:
        """Evaluate a single spread bet for value."""
        game_id = game.get("game_id")
        home_team = game.get("home_team", "")
        away_team = game.get("away_team", "")
        game_time = game.get("game_time")

        bookmaker = odds.get("bookmaker", "Unknown")
        spread = odds.get("home_spread", 0.0)
        home_odds = odds.get("home_odds", -110)
        away_odds = odds.get("away_odds", -110)

        if side == "home":
            line = spread
            market_odds = home_odds
            opp_odds = away_odds
            model_prob = prediction.home_cover_prob
            description = f"{home_team} {line:+.1f}"
        else:
            line = -spread
            market_odds = away_odds
            opp_odds = home_odds
            model_prob = 1 - prediction.home_cover_prob
            description = f"{away_team} {line:+.1f}"

        # Calculate edge
        edge, ev, raw_edge = self._calculate_edge_and_ev(
            model_prob=model_prob,
            odds=market_odds,
            opposite_odds=opp_odds,
        )

        # Check thresholds
        if edge < self.min_edge or ev < self.min_ev:
            return None

        # Determine urgency
        urgency = self._determine_urgency(edge, game_time)

        bet_id = f"{game_id}_{side}_spread_{bookmaker}"

        return ValueBet(
            bet_id=bet_id,
            bet_type=BetType.SPREAD,
            game_id=game_id,
            description=description,
            model_probability=model_prob,
            model_prediction=prediction.predicted_spread,
            bookmaker=bookmaker,
            odds=market_odds,
            implied_probability=float(american_to_implied_probability(market_odds)),
            line=line,
            edge=edge,
            expected_value=ev,
            raw_edge=raw_edge,
            urgency=urgency,
            home_team=home_team,
            away_team=away_team,
            game_time=game_time,
        )

    def scan_props(
        self,
        props: list[dict],
        player_features: dict[str, dict],
    ) -> DetectionResult:
        """
        Scan player props for value opportunities.

        Args:
            props: List of prop dicts with player_id, prop_type, line, over_odds, under_odds
            player_features: Dict mapping player_id to feature dict

        Returns:
            DetectionResult with detected value bets
        """
        value_bets = []

        for prop in props:
            player_id = prop.get("player_id")
            prop_type = prop.get("prop_type")

            if not player_id or not prop_type:
                continue

            # Get appropriate model
            model = self.prop_models.get(prop_type)
            if model is None:
                continue

            # Get features
            features = player_features.get(player_id)
            if features is None:
                continue

            # Evaluate over and under
            for side in ["over", "under"]:
                bet = self._evaluate_prop_bet(
                    model=model,
                    prop=prop,
                    features=features,
                    side=side,
                )
                if bet:
                    value_bets.append(bet)

        # Sort by edge
        value_bets.sort(key=lambda b: b.edge, reverse=True)

        return DetectionResult(
            value_bets=value_bets,
            props_scanned=len(props),
            total_opportunities=len(value_bets),
        )

    def _evaluate_prop_bet(
        self,
        model,
        prop: dict,
        features: dict,
        side: str,
    ) -> Optional[ValueBet]:
        """Evaluate a single prop bet for value."""
        player_id = prop.get("player_id")
        player_name = prop.get("player_name", "Unknown")
        game_id = prop.get("game_id", "")
        prop_type = prop.get("prop_type")
        line = prop.get("line", 0.0)
        over_odds = prop.get("over_odds", -110)
        under_odds = prop.get("under_odds", -110)
        bookmaker = prop.get("bookmaker", "Unknown")
        game_time = prop.get("game_time")
        team = prop.get("team", "")
        opponent = prop.get("opponent", "")

        try:
            # Get model prediction
            prediction = model.predict_player(
                features=features,
                player_id=player_id,
                player_name=player_name,
                game_id=game_id,
                team=team,
                opponent=opponent,
                line=line,
            )
        except Exception as e:
            logger.warning(f"Failed to predict {player_name} {prop_type}: {e}")
            return None

        if side == "over":
            market_odds = over_odds
            opp_odds = under_odds
            model_prob = prediction.over_prob or 0.5
            description = f"{player_name} Over {line} {prop_type}"
        else:
            market_odds = under_odds
            opp_odds = over_odds
            model_prob = prediction.under_prob or 0.5
            description = f"{player_name} Under {line} {prop_type}"

        # Calculate edge
        edge, ev, raw_edge = self._calculate_edge_and_ev(
            model_prob=model_prob,
            odds=market_odds,
            opposite_odds=opp_odds,
        )

        # Check thresholds
        if edge < self.min_edge or ev < self.min_ev:
            return None

        # Determine urgency
        urgency = self._determine_urgency(edge, game_time)

        bet_id = f"{game_id}_{player_id}_{prop_type}_{side}_{bookmaker}"

        return ValueBet(
            bet_id=bet_id,
            bet_type=BetType(prop_type) if prop_type in BetType.__members__.values() else BetType.PASSING_YARDS,
            game_id=game_id,
            description=description,
            model_probability=model_prob,
            model_prediction=prediction.predicted_value,
            bookmaker=bookmaker,
            odds=market_odds,
            implied_probability=float(american_to_implied_probability(market_odds)),
            line=line,
            edge=edge,
            expected_value=ev,
            raw_edge=raw_edge,
            urgency=urgency,
            player_name=player_name,
            player_id=player_id,
            game_time=game_time,
        )

    def _calculate_edge_and_ev(
        self,
        model_prob: float,
        odds: int,
        opposite_odds: int,
    ) -> tuple[float, float, float]:
        """
        Calculate edge and expected value.

        Args:
            model_prob: Model's win probability
            odds: American odds for this side
            opposite_odds: American odds for opposite side

        Returns:
            Tuple of (edge, ev, raw_edge)
        """
        model_prob_decimal = Decimal(str(model_prob))

        # Raw edge (vs implied with vig)
        raw_implied = american_to_implied_probability(odds)
        raw_edge = float(model_prob_decimal - raw_implied)

        # Edge vs no-vig probability
        if self.remove_vig:
            fair_prob, _ = remove_vig(odds, opposite_odds)
            edge = float(model_prob_decimal - fair_prob)
        else:
            edge = raw_edge

        # Expected value
        ev = float(calculate_expected_value(model_prob_decimal, odds))

        return edge, ev, raw_edge

    def _determine_urgency(
        self,
        edge: float,
        game_time: Optional[datetime],
    ) -> Urgency:
        """
        Determine urgency level based on edge and time to game.

        Args:
            edge: Edge percentage
            game_time: Game start time

        Returns:
            Urgency level
        """
        if game_time:
            hours_until = (game_time - datetime.now()).total_seconds() / 3600
        else:
            hours_until = 24  # Default to 24 hours if unknown

        # CRITICAL: High edge + close to game time
        if edge >= self.CRITICAL_EDGE and hours_until <= self.CRITICAL_HOURS:
            return Urgency.CRITICAL

        # CRITICAL: Very high edge regardless of time
        if edge >= self.CRITICAL_EDGE:
            return Urgency.HIGH

        # HIGH: Moderate edge + close to game time
        if edge >= self.HIGH_EDGE or hours_until <= self.HIGH_HOURS:
            return Urgency.HIGH

        # MEDIUM: Meets thresholds
        if edge >= self.min_edge:
            return Urgency.MEDIUM

        return Urgency.LOW

    def find_best_books(
        self,
        game_id: str,
        side: str,
        all_odds: list[dict],
    ) -> list[dict]:
        """
        Find the best odds across bookmakers for a given side.

        Args:
            game_id: Game identifier
            side: "home" or "away"
            all_odds: List of odds from all bookmakers

        Returns:
            List of dicts with bookmaker and odds, sorted best first
        """
        game_odds = [o for o in all_odds if o.get("game_id") == game_id]

        if side == "home":
            odds_key = "home_odds"
        else:
            odds_key = "away_odds"

        results = [
            {"bookmaker": o.get("bookmaker", "Unknown"), "odds": o.get(odds_key, -110)}
            for o in game_odds
            if odds_key in o
        ]

        # Sort by best odds (highest for positive, highest absolute for negative)
        results.sort(key=lambda x: american_to_decimal(x["odds"]), reverse=True)

        return results


def detect_value_bets(
    spread_model,
    games: list[dict],
    odds_data: list[dict],
    features: dict[str, dict],
    min_edge: float = 0.03,
    min_ev: float = 0.02,
) -> list[ValueBet]:
    """
    Convenience function to detect spread value bets.

    Args:
        spread_model: Trained SpreadModel
        games: List of game information
        odds_data: List of odds from bookmakers
        features: Feature dict for each game
        min_edge: Minimum edge threshold
        min_ev: Minimum EV threshold

    Returns:
        List of ValueBet opportunities
    """
    detector = ValueDetector(
        spread_model=spread_model,
        min_edge=min_edge,
        min_ev=min_ev,
    )
    result = detector.scan_spreads(games, odds_data, features)
    return result.value_bets
