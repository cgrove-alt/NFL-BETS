"""
Odds conversion and calculation utilities.

Provides functions for converting between odds formats and calculating
implied probabilities, expected value, and vig.
"""
from decimal import Decimal, ROUND_HALF_UP
from typing import NamedTuple


class OddsFormats(NamedTuple):
    """Container for odds in multiple formats."""

    american: int
    decimal: Decimal
    implied_probability: Decimal


class VigorousLine(NamedTuple):
    """Two-way line with vig information."""

    side1_implied: Decimal
    side2_implied: Decimal
    total_implied: Decimal
    vig_percent: Decimal
    side1_fair: Decimal
    side2_fair: Decimal


def american_to_decimal(american: int) -> Decimal:
    """
    Convert American odds to decimal odds.

    Args:
        american: American odds (e.g., -110, +150)

    Returns:
        Decimal odds (e.g., 1.91, 2.50)

    Examples:
        >>> american_to_decimal(-110)
        Decimal('1.909090909090909090909090909')
        >>> american_to_decimal(+150)
        Decimal('2.5')
    """
    if american > 0:
        return Decimal(american) / Decimal("100") + Decimal("1")
    else:
        return Decimal("100") / Decimal(abs(american)) + Decimal("1")


def decimal_to_american(decimal_odds: Decimal) -> int:
    """
    Convert decimal odds to American odds.

    Args:
        decimal_odds: Decimal odds (e.g., 1.91, 2.50)

    Returns:
        American odds (e.g., -110, +150)

    Examples:
        >>> decimal_to_american(Decimal('1.91'))
        -110
        >>> decimal_to_american(Decimal('2.50'))
        150
    """
    if decimal_odds >= 2:
        # Positive American odds
        return int(((decimal_odds - 1) * 100).quantize(Decimal("1")))
    else:
        # Negative American odds
        return int((-100 / (decimal_odds - 1)).quantize(Decimal("1")))


def american_to_implied_probability(american: int) -> Decimal:
    """
    Convert American odds to implied probability.

    Note: This includes the bookmaker's vig, so probabilities won't sum to 1.

    Args:
        american: American odds

    Returns:
        Implied probability (0-1)

    Examples:
        >>> american_to_implied_probability(-110)
        Decimal('0.5238095238095238095238095238')
        >>> american_to_implied_probability(+150)
        Decimal('0.4')
    """
    if american > 0:
        return Decimal("100") / (Decimal(american) + Decimal("100"))
    else:
        return Decimal(abs(american)) / (Decimal(abs(american)) + Decimal("100"))


def decimal_to_implied_probability(decimal_odds: Decimal) -> Decimal:
    """
    Convert decimal odds to implied probability.

    Args:
        decimal_odds: Decimal odds

    Returns:
        Implied probability (0-1)

    Examples:
        >>> decimal_to_implied_probability(Decimal('2.0'))
        Decimal('0.5')
    """
    return Decimal("1") / decimal_odds


def implied_probability_to_american(probability: Decimal) -> int:
    """
    Convert implied probability to American odds.

    Args:
        probability: Implied probability (0-1)

    Returns:
        American odds

    Examples:
        >>> implied_probability_to_american(Decimal('0.5'))
        100
        >>> implied_probability_to_american(Decimal('0.6'))
        -150
    """
    if probability == Decimal("0.5"):
        return 100
    elif probability > Decimal("0.5"):
        # Favorite (negative odds)
        return int((-probability / (1 - probability) * 100).quantize(Decimal("1")))
    else:
        # Underdog (positive odds)
        return int(((1 - probability) / probability * 100).quantize(Decimal("1")))


def convert_odds(american: int) -> OddsFormats:
    """
    Convert American odds to all formats.

    Args:
        american: American odds

    Returns:
        OddsFormats with american, decimal, and implied probability
    """
    return OddsFormats(
        american=american,
        decimal=american_to_decimal(american),
        implied_probability=american_to_implied_probability(american),
    )


def calculate_vig(odds1: int, odds2: int) -> VigorousLine:
    """
    Calculate the vig/juice for a two-way line.

    Args:
        odds1: American odds for outcome 1
        odds2: American odds for outcome 2

    Returns:
        VigorousLine with implied probabilities, vig, and fair probabilities

    Examples:
        >>> result = calculate_vig(-110, -110)
        >>> result.vig_percent
        Decimal('4.761904761904761904761904762')
    """
    implied1 = american_to_implied_probability(odds1)
    implied2 = american_to_implied_probability(odds2)

    total_implied = implied1 + implied2

    # Vig is the excess over 100%
    vig_percent = (total_implied - Decimal("1")) * Decimal("100")

    # Fair (no-vig) probabilities
    fair1 = implied1 / total_implied
    fair2 = implied2 / total_implied

    return VigorousLine(
        side1_implied=implied1,
        side2_implied=implied2,
        total_implied=total_implied,
        vig_percent=vig_percent,
        side1_fair=fair1,
        side2_fair=fair2,
    )


def remove_vig(odds1: int, odds2: int) -> tuple[Decimal, Decimal]:
    """
    Remove vig to get fair/true probabilities.

    Uses the multiplicative method.

    Args:
        odds1: American odds for outcome 1
        odds2: American odds for outcome 2

    Returns:
        Tuple of (fair_prob1, fair_prob2) summing to 1.0
    """
    result = calculate_vig(odds1, odds2)
    return (result.side1_fair, result.side2_fair)


def calculate_expected_value(
    win_probability: Decimal,
    american_odds: int,
) -> Decimal:
    """
    Calculate expected value per unit wagered.

    EV = (probability * win_amount) - ((1 - probability) * stake)
    For unit stake at decimal odds d: EV = p * (d - 1) - (1 - p)

    Args:
        win_probability: Model's estimated probability of winning (0-1)
        american_odds: American odds offered

    Returns:
        Expected value per $1 wagered (positive = profitable)

    Examples:
        >>> calculate_expected_value(Decimal('0.55'), -110)
        Decimal('0.04545454545454545454545454545')  # ~4.5% EV
    """
    decimal_odds = american_to_decimal(american_odds)

    # EV = p * (d - 1) - (1 - p)
    # = p*d - p - 1 + p
    # = p*d - 1
    ev = win_probability * decimal_odds - Decimal("1")

    return ev


def calculate_edge(
    model_probability: Decimal,
    american_odds: int,
) -> Decimal:
    """
    Calculate edge over the market.

    Edge = Model Probability - Implied Probability

    Args:
        model_probability: Model's estimated probability (0-1)
        american_odds: American odds offered

    Returns:
        Edge as decimal (positive = value bet)

    Examples:
        >>> calculate_edge(Decimal('0.55'), -110)
        Decimal('0.02619047619047619047619047620')  # ~2.6% edge
    """
    implied_probability = american_to_implied_probability(american_odds)
    return model_probability - implied_probability


def calculate_break_even_probability(american_odds: int) -> Decimal:
    """
    Calculate the break-even win rate for given odds.

    This is the minimum win probability needed to profit long-term.

    Args:
        american_odds: American odds

    Returns:
        Break-even probability (0-1)

    Examples:
        >>> calculate_break_even_probability(-110)
        Decimal('0.5238095238095238095238095238')  # 52.38%
    """
    return american_to_implied_probability(american_odds)


def calculate_fair_odds(probability: Decimal) -> int:
    """
    Calculate what the fair odds should be for a given probability.

    Fair odds have no vig - what the line would be if bookmaker took no margin.

    Args:
        probability: True probability of outcome (0-1)

    Returns:
        Fair American odds
    """
    return implied_probability_to_american(probability)


def calculate_potential_payout(stake: Decimal, american_odds: int) -> Decimal:
    """
    Calculate total payout (stake + profit) for a winning bet.

    Args:
        stake: Amount wagered
        american_odds: American odds

    Returns:
        Total payout if bet wins

    Examples:
        >>> calculate_potential_payout(Decimal('100'), -110)
        Decimal('190.90909090909090909090909091')
    """
    decimal_odds = american_to_decimal(american_odds)
    return stake * decimal_odds


def calculate_profit(stake: Decimal, american_odds: int) -> Decimal:
    """
    Calculate profit for a winning bet.

    Args:
        stake: Amount wagered
        american_odds: American odds

    Returns:
        Profit if bet wins (excludes returned stake)

    Examples:
        >>> calculate_profit(Decimal('100'), -110)
        Decimal('90.90909090909090909090909091')
        >>> calculate_profit(Decimal('100'), +150)
        Decimal('150')
    """
    return calculate_potential_payout(stake, american_odds) - stake


def format_american_odds(odds: int) -> str:
    """
    Format American odds with proper sign.

    Args:
        odds: American odds

    Returns:
        Formatted string with +/- prefix

    Examples:
        >>> format_american_odds(-110)
        '-110'
        >>> format_american_odds(150)
        '+150'
    """
    if odds > 0:
        return f"+{odds}"
    return str(odds)


def format_probability_percent(probability: Decimal, decimals: int = 1) -> str:
    """
    Format probability as percentage string.

    Args:
        probability: Probability as decimal (0-1)
        decimals: Number of decimal places

    Returns:
        Formatted percentage string

    Examples:
        >>> format_probability_percent(Decimal('0.5238'))
        '52.4%'
    """
    percent = probability * 100
    return f"{percent:.{decimals}f}%"


def check_arbitrage(odds1: int, odds2: int) -> tuple[bool, Decimal]:
    """
    Check if arbitrage opportunity exists between two sides.

    Arbitrage exists when total implied probability < 100%.

    Args:
        odds1: American odds for outcome 1 (from bookmaker A)
        odds2: American odds for outcome 2 (from bookmaker B)

    Returns:
        Tuple of (is_arbitrage, profit_percent)

    Examples:
        >>> check_arbitrage(-102, 101)  # Slight arb
        (True, Decimal('...'))
        >>> check_arbitrage(-110, -110)  # No arb
        (False, Decimal('-4.761904761904761904761904762'))
    """
    implied1 = american_to_implied_probability(odds1)
    implied2 = american_to_implied_probability(odds2)

    total_implied = implied1 + implied2

    # Profit percentage = (1 - total_implied) / total_implied * 100
    if total_implied < Decimal("1"):
        profit_pct = (Decimal("1") - total_implied) / total_implied * 100
        return (True, profit_pct)
    else:
        # Negative value shows how much we'd lose
        loss_pct = (total_implied - Decimal("1")) / total_implied * 100
        return (False, -loss_pct)


def calculate_arbitrage_stakes(
    odds1: int,
    odds2: int,
    total_stake: Decimal,
) -> tuple[Decimal, Decimal, Decimal]:
    """
    Calculate optimal stake distribution for arbitrage.

    Args:
        odds1: American odds for outcome 1
        odds2: American odds for outcome 2
        total_stake: Total amount to distribute

    Returns:
        Tuple of (stake1, stake2, guaranteed_profit)

    Examples:
        >>> calculate_arbitrage_stakes(-102, 101, Decimal('1000'))
        (Decimal('...'), Decimal('...'), Decimal('...'))
    """
    implied1 = american_to_implied_probability(odds1)
    implied2 = american_to_implied_probability(odds2)

    total_implied = implied1 + implied2

    # Allocate stakes proportionally
    stake1 = (total_stake * implied1 / total_implied).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    )
    stake2 = (total_stake * implied2 / total_implied).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    )

    # Calculate returns
    decimal1 = american_to_decimal(odds1)
    decimal2 = american_to_decimal(odds2)

    return1 = stake1 * decimal1
    return2 = stake2 * decimal2

    # Guaranteed profit is the minimum return minus total stake
    guaranteed_profit = min(return1, return2) - total_stake

    return (stake1, stake2, guaranteed_profit)
