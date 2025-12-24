"""
Kelly Criterion bet sizing calculator.

Implements optimal bet sizing using the Kelly Criterion with
conservative fractional Kelly to reduce variance while maintaining
positive expected growth.
"""

from dataclasses import dataclass, field
from typing import Optional

from .odds_converter import american_to_decimal as _american_to_decimal


def american_to_decimal(american: int) -> float:
    """Convert American odds to decimal odds as float."""
    return float(_american_to_decimal(american))


@dataclass
class StakeRecommendation:
    """Recommended stake for a single bet."""

    bet_id: str
    full_kelly: float  # Full Kelly fraction (0-1)
    fractional_kelly: float  # After applying fraction (e.g., 25%)
    recommended_stake: float  # Dollar amount
    stake_percentage: float  # Percentage of bankroll

    # Constraints applied
    capped_by_max: bool = False
    capped_by_min: bool = False
    below_minimum: bool = False  # Stake too small to place

    # Input parameters for reference
    win_probability: float = 0.0
    decimal_odds: float = 0.0
    bankroll: float = 0.0


@dataclass
class SimultaneousStakes:
    """Stakes for multiple simultaneous bets."""

    stakes: dict[str, StakeRecommendation] = field(default_factory=dict)
    total_exposure: float = 0.0
    exposure_percentage: float = 0.0
    adjusted_for_correlation: bool = False


class KellyCalculator:
    """
    Kelly Criterion calculator for optimal bet sizing.

    The Kelly Criterion provides mathematically optimal bet sizes
    to maximize long-term bankroll growth. However, full Kelly
    can be volatile, so we use fractional Kelly (25% by default).

    Key formulas:
    - Full Kelly: f* = (bp - q) / b
      where b = net odds, p = win prob, q = lose prob
    - Fractional Kelly: f = f* × fraction

    Example:
        >>> kelly = KellyCalculator(fraction=0.25, max_stake_pct=0.05)
        >>> stake = kelly.calculate_stake(
        ...     bankroll=1000.0,
        ...     win_probability=0.55,
        ...     odds=-110
        ... )
        >>> print(f"Recommended stake: ${stake.recommended_stake:.2f}")
    """

    def __init__(
        self,
        fraction: float = 0.25,
        max_stake_pct: float = 0.05,
        min_stake: float = 10.0,
        max_daily_exposure: float = 0.20,
    ):
        """
        Initialize the Kelly calculator.

        Args:
            fraction: Kelly fraction to use (0.25 = quarter Kelly)
            max_stake_pct: Maximum stake as percentage of bankroll
            min_stake: Minimum dollar amount for a bet
            max_daily_exposure: Maximum total exposure as % of bankroll
        """
        if not 0 < fraction <= 1:
            raise ValueError("Fraction must be between 0 and 1")
        if not 0 < max_stake_pct <= 1:
            raise ValueError("Max stake percentage must be between 0 and 1")

        self.fraction = fraction
        self.max_stake_pct = max_stake_pct
        self.min_stake = min_stake
        self.max_daily_exposure = max_daily_exposure

    def full_kelly(
        self,
        win_probability: float,
        decimal_odds: float,
    ) -> float:
        """
        Calculate full Kelly stake fraction.

        Args:
            win_probability: Probability of winning (0-1)
            decimal_odds: Decimal odds (e.g., 1.91 for -110)

        Returns:
            Kelly fraction (0-1), or 0 if negative edge
        """
        if not 0 < win_probability < 1:
            return 0.0
        if decimal_odds <= 1:
            return 0.0

        b = decimal_odds - 1  # Net odds (profit per unit wagered)
        p = win_probability
        q = 1 - p

        # Kelly formula: f* = (bp - q) / b
        kelly = (b * p - q) / b

        # Can't bet negative or more than 100%
        return max(0.0, min(kelly, 1.0))

    def fractional_kelly(
        self,
        win_probability: float,
        decimal_odds: float,
    ) -> float:
        """
        Calculate fractional Kelly stake fraction.

        Args:
            win_probability: Probability of winning (0-1)
            decimal_odds: Decimal odds (e.g., 1.91 for -110)

        Returns:
            Fractional Kelly fraction (0-1)
        """
        full = self.full_kelly(win_probability, decimal_odds)
        return full * self.fraction

    def calculate_stake(
        self,
        bankroll: float,
        win_probability: float,
        odds: int,
        bet_id: str = "",
    ) -> StakeRecommendation:
        """
        Calculate recommended stake for a bet.

        Args:
            bankroll: Current bankroll in dollars
            win_probability: Probability of winning (0-1)
            odds: American odds (e.g., -110, +150)
            bet_id: Optional identifier for the bet

        Returns:
            StakeRecommendation with dollar amount and metadata
        """
        decimal_odds = american_to_decimal(odds)

        # Calculate Kelly fractions
        full = self.full_kelly(win_probability, decimal_odds)
        fractional = full * self.fraction

        # Calculate dollar stake
        stake = fractional * bankroll

        # Apply constraints
        capped_by_max = False
        capped_by_min = False
        below_minimum = False

        max_stake = self.max_stake_pct * bankroll

        if stake > max_stake:
            stake = max_stake
            capped_by_max = True

        if stake < self.min_stake:
            if stake > 0:
                # Check if we should round up to minimum
                if stake >= self.min_stake * 0.5:
                    stake = self.min_stake
                    capped_by_min = True
                else:
                    # Edge too small, don't bet
                    below_minimum = True
                    stake = 0.0
            else:
                below_minimum = True

        stake_percentage = stake / bankroll if bankroll > 0 else 0

        return StakeRecommendation(
            bet_id=bet_id,
            full_kelly=full,
            fractional_kelly=fractional,
            recommended_stake=round(stake, 2),
            stake_percentage=stake_percentage,
            capped_by_max=capped_by_max,
            capped_by_min=capped_by_min,
            below_minimum=below_minimum,
            win_probability=win_probability,
            decimal_odds=decimal_odds,
            bankroll=bankroll,
        )

    def calculate_simultaneous_kelly(
        self,
        bets: list[dict],
        bankroll: float,
        current_exposure: float = 0.0,
        correlation_factor: float = 0.8,
    ) -> SimultaneousStakes:
        """
        Calculate stakes for multiple simultaneous bets.

        When placing multiple bets at once, we need to:
        1. Reduce individual stakes to stay under daily exposure limit
        2. Account for potential correlation between bets
        3. Ensure total exposure doesn't exceed limits

        Args:
            bets: List of dicts with 'bet_id', 'win_probability', 'odds'
            bankroll: Current bankroll
            current_exposure: Amount already at risk today
            correlation_factor: Reduction factor for correlated bets (0-1)

        Returns:
            SimultaneousStakes with individual stake recommendations
        """
        if not bets:
            return SimultaneousStakes()

        # Calculate individual Kelly stakes first
        individual_stakes: dict[str, StakeRecommendation] = {}
        total_individual = 0.0

        for bet in bets:
            stake_rec = self.calculate_stake(
                bankroll=bankroll,
                win_probability=bet["win_probability"],
                odds=bet["odds"],
                bet_id=bet.get("bet_id", ""),
            )
            individual_stakes[stake_rec.bet_id] = stake_rec
            total_individual += stake_rec.recommended_stake

        # Calculate available exposure
        max_exposure = self.max_daily_exposure * bankroll
        available_exposure = max_exposure - current_exposure

        if available_exposure <= 0:
            # Already at daily limit
            return SimultaneousStakes(
                stakes={
                    k: StakeRecommendation(
                        bet_id=k,
                        full_kelly=v.full_kelly,
                        fractional_kelly=0,
                        recommended_stake=0,
                        stake_percentage=0,
                        below_minimum=True,
                        win_probability=v.win_probability,
                        decimal_odds=v.decimal_odds,
                        bankroll=bankroll,
                    )
                    for k, v in individual_stakes.items()
                },
                total_exposure=current_exposure,
                exposure_percentage=current_exposure / bankroll,
            )

        # Apply correlation adjustment and exposure limits
        adjusted_stakes: dict[str, StakeRecommendation] = {}
        adjusted_for_correlation = False

        if total_individual > available_exposure:
            # Scale down proportionally and apply correlation factor
            scale = (available_exposure / total_individual) * correlation_factor
            adjusted_for_correlation = True

            for bet_id, stake_rec in individual_stakes.items():
                new_stake = stake_rec.recommended_stake * scale

                # Apply minimum constraint
                below_min = False
                if new_stake < self.min_stake:
                    if new_stake >= self.min_stake * 0.5:
                        new_stake = self.min_stake
                    else:
                        new_stake = 0.0
                        below_min = True

                adjusted_stakes[bet_id] = StakeRecommendation(
                    bet_id=bet_id,
                    full_kelly=stake_rec.full_kelly,
                    fractional_kelly=stake_rec.fractional_kelly * scale,
                    recommended_stake=round(new_stake, 2),
                    stake_percentage=new_stake / bankroll,
                    capped_by_max=True,
                    below_minimum=below_min,
                    win_probability=stake_rec.win_probability,
                    decimal_odds=stake_rec.decimal_odds,
                    bankroll=bankroll,
                )
        else:
            # No scaling needed, use individual stakes
            adjusted_stakes = individual_stakes

        total_exposure = sum(s.recommended_stake for s in adjusted_stakes.values())

        return SimultaneousStakes(
            stakes=adjusted_stakes,
            total_exposure=round(total_exposure + current_exposure, 2),
            exposure_percentage=(total_exposure + current_exposure) / bankroll,
            adjusted_for_correlation=adjusted_for_correlation,
        )

    def calculate_growth_rate(
        self,
        win_probability: float,
        decimal_odds: float,
        stake_fraction: Optional[float] = None,
    ) -> float:
        """
        Calculate expected logarithmic growth rate.

        The Kelly Criterion maximizes this quantity, which represents
        the expected log-growth of the bankroll.

        Args:
            win_probability: Probability of winning
            decimal_odds: Decimal odds
            stake_fraction: Fraction of bankroll to wager (uses Kelly if None)

        Returns:
            Expected log growth rate
        """
        if stake_fraction is None:
            stake_fraction = self.fractional_kelly(win_probability, decimal_odds)

        if stake_fraction <= 0:
            return 0.0

        p = win_probability
        q = 1 - p
        b = decimal_odds - 1

        # Expected log growth: p * log(1 + bf) + q * log(1 - f)
        import math

        try:
            win_term = p * math.log(1 + b * stake_fraction)
            lose_term = q * math.log(1 - stake_fraction)
            return win_term + lose_term
        except ValueError:
            return float("-inf")

    def optimal_fraction_for_target_risk(
        self,
        win_probability: float,
        decimal_odds: float,
        target_risk: float = 0.05,
    ) -> float:
        """
        Find stake fraction that achieves target risk level.

        Rather than using a fixed Kelly fraction, find the stake
        that corresponds to a specific probability of ruin.

        Args:
            win_probability: Probability of winning
            decimal_odds: Decimal odds
            target_risk: Target probability of losing entire stake

        Returns:
            Optimal stake fraction
        """
        full = self.full_kelly(win_probability, decimal_odds)

        if full <= 0:
            return 0.0

        # For target risk, we typically use a more conservative approach
        # Approximate: if we bet f of bankroll, risk of ruin after n bets
        # is roughly (q/p)^(1/f) for p > q

        # Simple approximation: use fractional Kelly scaled by risk tolerance
        risk_scale = min(target_risk / 0.10, 1.0)  # 10% risk = full fraction
        return full * self.fraction * risk_scale


def calculate_optimal_kelly(
    win_probability: float,
    american_odds: int,
    fraction: float = 0.25,
) -> float:
    """
    Convenience function to calculate fractional Kelly stake.

    Args:
        win_probability: Probability of winning (0-1)
        american_odds: American odds (e.g., -110)
        fraction: Kelly fraction (default 25%)

    Returns:
        Recommended stake as fraction of bankroll
    """
    decimal_odds = american_to_decimal(american_odds)
    b = decimal_odds - 1
    p = win_probability
    q = 1 - p

    if b <= 0 or p <= 0 or p >= 1:
        return 0.0

    kelly = (b * p - q) / b
    return max(0.0, kelly * fraction)
