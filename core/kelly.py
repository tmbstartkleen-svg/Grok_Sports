"""
Kelly Criterion and fractional Kelly bankroll sizing.
Also supports multi-bet portfolio Kelly approximation.
"""

from __future__ import annotations
import numpy as np
from typing import Sequence, Tuple


def kelly_fraction(p: float, decimal_odds: float) -> float:
    """
    Full Kelly fraction for a single bet.
    f* = (p*(b+1) - 1) / b   where b = decimal_odds - 1
    Returns 0 if edge is non-positive.
    """
    if decimal_odds <= 1.0 or p <= 0.0 or p >= 1.0:
        return 0.0
    b = decimal_odds - 1.0
    edge = p * (b + 1.0) - 1.0
    if edge <= 0:
        return 0.0
    return edge / b


def fractional_kelly(p: float, decimal_odds: float, fraction: float = 0.5) -> float:
    """Half-Kelly (or other fraction) for safety against model error."""
    return max(0.0, kelly_fraction(p, decimal_odds) * fraction)


def kelly_stake(bankroll: float, p: float, decimal_odds: float, fraction: float = 0.5,
                max_pct: float = 0.05) -> float:
    """
    Dollar stake = bankroll * min(fractional_kelly, max_pct).
    Caps exposure per bet.
    """
    f = fractional_kelly(p, decimal_odds, fraction)
    f = min(f, max_pct)
    return bankroll * f


def portfolio_kelly_approx(
    ps: Sequence[float],
    decimal_odds: Sequence[float],
    correlations: np.ndarray | None = None,
    fraction: float = 0.25,
) -> np.ndarray:
    """
    Approximate joint Kelly for a set of simultaneous bets.
    Uses the independent solution then shrinks by average pairwise correlation
    and overall fraction. For production use a proper quadratic solver.
    """
    ps = np.asarray(ps, dtype=float)
    odds = np.asarray(decimal_odds, dtype=float)
    n = len(ps)
    raw = np.array([kelly_fraction(ps[i], odds[i]) for i in range(n)])
    if correlations is None:
        shrink = 1.0
    else:
        # average off-diagonal correlation
        mask = ~np.eye(n, dtype=bool)
        avg_corr = float(np.mean(np.abs(correlations[mask]))) if n > 1 else 0.0
        shrink = max(0.2, 1.0 - 0.7 * avg_corr)
    return np.maximum(0.0, raw * fraction * shrink)
