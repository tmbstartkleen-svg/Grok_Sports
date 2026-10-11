"""
Expected value, implied probability, vig removal, ranking.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List, Optional


def american_to_decimal(american: float) -> float:
    if american > 0:
        return 1.0 + american / 100.0
    else:
        return 1.0 + 100.0 / abs(american)


def decimal_to_american(decimal: float) -> float:
    if decimal >= 2.0:
        return (decimal - 1.0) * 100.0
    else:
        return -100.0 / (decimal - 1.0)


def implied_prob(decimal_odds: float) -> float:
    return 1.0 / decimal_odds


def remove_vig_multiplicative(odds_list: List[float]) -> List[float]:
    """Simple multiplicative vig removal for a market (e.g. two-way ML)."""
    imps = [implied_prob(o) for o in odds_list]
    total = sum(imps)
    return [imp / total for imp in imps]


def expected_value(p: float, decimal_odds: float, stake: float = 1.0) -> float:
    """EV of a unit (or given) stake."""
    return stake * (p * (decimal_odds - 1.0) - (1.0 - p))


def edge(p: float, decimal_odds: float) -> float:
    """p - implied probability."""
    return p - implied_prob(decimal_odds)


@dataclass
class Leg:
    id: str
    description: str
    p: float                  # true probability from model / Monte Carlo
    decimal_odds: float
    american_odds: float
    edge: float
    ev: float                 # EV per $1 stake
    ci_low: float = 0.0
    ci_high: float = 1.0
    sport: str = ""
    start_time: str = ""      # ISO or day offset
    correlation_group: str = ""  # e.g. same game id for correlation handling


def rank_legs(legs: List[Leg], min_edge: float = 0.02, min_p: float = 0.45) -> List[Leg]:
    """Filter and rank by EV descending, then edge."""
    filtered = [l for l in legs if l.edge >= min_edge and l.p >= min_p and l.ev > 0]
    return sorted(filtered, key=lambda x: (x.ev, x.edge), reverse=True)
