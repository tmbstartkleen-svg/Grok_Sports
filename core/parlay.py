"""
Parlay construction with combined probability constraint and correlation awareness.
"""

from __future__ import annotations
import itertools
import numpy as np
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple, Optional
from .ev import Leg, expected_value, implied_prob


@dataclass
class Parlay:
    legs: List[Leg]
    combined_p: float
    decimal_odds: float
    american_odds: float
    ev: float
    n_legs: int
    correlation_penalty: float = 0.0
    description: str = ""

    def __post_init__(self):
        self.n_legs = len(self.legs)
        if not self.description:
            self.description = " + ".join(l.description for l in self.legs)


def independent_combined_p(legs: Sequence[Leg]) -> float:
    p = 1.0
    for leg in legs:
        p *= leg.p
    return p


def independent_decimal_odds(legs: Sequence[Leg]) -> float:
    odds = 1.0
    for leg in legs:
        odds *= leg.decimal_odds
    return odds


def correlation_adjusted_p(legs: Sequence[Leg], avg_pairwise_corr: float = 0.15) -> float:
    """
    Crude but useful adjustment: independence overestimates joint probability
    when legs are positively correlated. Shrink toward the minimum p.
    """
    if len(legs) == 1:
        return legs[0].p
    indep = independent_combined_p(legs)
    min_p = min(l.p for l in legs)
    # convex combination controlled by correlation strength
    return (1.0 - avg_pairwise_corr) * indep + avg_pairwise_corr * (min_p * indep ** 0.5)


def build_parlays(
    candidate_legs: List[Leg],
    max_legs: int = 4,
    min_combined_p: float = 0.52,
    min_ev: float = 0.0,
    max_parlays: int = 50,
    prefer_short: bool = True,
) -> List[Parlay]:
    """
    Enumerate combinations up to max_legs, keep those with
    combined_p >= min_combined_p and positive EV.
    """
    parlays: List[Parlay] = []
    n = len(candidate_legs)
    for k in range(2, min(max_legs, n) + 1):
        for combo in itertools.combinations(candidate_legs, k):
            # skip if any two share the same correlation_group (same-game)
            groups = [l.correlation_group for l in combo if l.correlation_group]
            if len(groups) != len(set(groups)):
                # same-game legs present → apply stronger correlation penalty
                adj_p = correlation_adjusted_p(combo, avg_pairwise_corr=0.35)
            else:
                adj_p = correlation_adjusted_p(combo, avg_pairwise_corr=0.12)

            if adj_p < min_combined_p:
                continue
            dec_odds = independent_decimal_odds(combo)
            ev = expected_value(adj_p, dec_odds)
            if ev < min_ev:
                continue
            am_odds = (dec_odds - 1.0) * 100 if dec_odds >= 2 else -100 / (dec_odds - 1.0)
            parlays.append(Parlay(
                legs=list(combo),
                combined_p=adj_p,
                decimal_odds=dec_odds,
                american_odds=am_odds,
                ev=ev,
                n_legs=k,
            ))
    # rank: prefer higher EV, then higher combined_p, then shorter if prefer_short
    def key(p: Parlay):
        short_bonus = -p.n_legs if prefer_short else 0
        return (p.ev, p.combined_p, short_bonus)
    parlays.sort(key=key, reverse=True)
    return parlays[:max_parlays]
