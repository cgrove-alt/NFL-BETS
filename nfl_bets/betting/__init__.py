"""
Betting utilities and value detection.

Provides tools for:
- Odds conversion and vig calculation
- Value bet detection with edge calculation
- Kelly Criterion bet sizing
- Cross-book arbitrage scanning
"""

from .odds_converter import (
    american_to_decimal,
    american_to_implied_probability,
    decimal_to_american,
    implied_probability_to_american,
    calculate_vig,
    remove_vig,
    calculate_edge,
    calculate_expected_value,
    check_arbitrage,
    calculate_arbitrage_stakes,
)

from .value_detector import (
    ValueBet,
    ValueDetector,
    DetectionResult,
    Urgency,
)

from .kelly_calculator import (
    KellyCalculator,
    StakeRecommendation,
    SimultaneousStakes,
    calculate_optimal_kelly,
)

from .arbitrage_scanner import (
    ArbitrageScanner,
    ArbitrageOpportunity,
    ArbType,
    BestLine,
    ScanResult,
)

__all__ = [
    # Odds converter
    "american_to_decimal",
    "american_to_implied_probability",
    "decimal_to_american",
    "implied_probability_to_american",
    "calculate_vig",
    "remove_vig",
    "calculate_edge",
    "calculate_expected_value",
    "check_arbitrage",
    "calculate_arbitrage_stakes",
    # Value detection
    "ValueBet",
    "ValueDetector",
    "DetectionResult",
    "Urgency",
    # Kelly sizing
    "KellyCalculator",
    "StakeRecommendation",
    "SimultaneousStakes",
    "calculate_optimal_kelly",
    # Arbitrage
    "ArbitrageScanner",
    "ArbitrageOpportunity",
    "ArbType",
    "BestLine",
    "ScanResult",
]
