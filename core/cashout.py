"""
Early cash-out / settlement EV calculator.

Given remaining legs, current offered cash-out value, and updated probabilities,
decide whether locking the cash-out has higher EV than letting the position ride.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, Optional
import numpy as np
from .correlation import CorrelationMatrix, correlation_adjusted_joint_prob


@dataclass
class CashoutDecision:
    ride_ev: float              # expected value if we keep the bet open
    cashout_value: float        # certain amount offered now
    cashout_ev: float           # = cashout_value (certain)
    recommend_cashout: bool
    edge_of_riding: float       # ride_ev - cashout_value
    remaining_combined_p: float
    fair_remaining_odds: float


def remaining_parlay_value(
    remaining_probs: Sequence[float],
    original_decimal_odds: float,
    original_stake: float,
    legs_already_won: int = 0,
    total_legs: Optional[int] = None,
    corr: Optional[CorrelationMatrix] = None,
) -> float:
    """
    Theoretical fair value of the open parlay = stake * original_odds * P(rest win).
    (Simplification: assumes equal contribution; real books use their own models.)
    """
    if not remaining_probs:
        return original_stake * original_decimal_odds  # already won
    if corr is not None and len(remaining_probs) > 1:
        p_rest = correlation_adjusted_joint_prob(remaining_probs, corr)
    else:
        p_rest = float(np.prod(remaining_probs))
    return original_stake * original_decimal_odds * p_rest


def evaluate_cashout(
    remaining_probs: Sequence[float],
    original_decimal_odds: float,
    original_stake: float,
    offered_cashout: float,
    corr: Optional[CorrelationMatrix] = None,
    risk_aversion: float = 0.0,
) -> CashoutDecision:
    """
    Compare offered cash-out to model fair value of riding.
    risk_aversion > 0 tilts toward cashing (utility adjustment).
    """
    fair = remaining_parlay_value(
        remaining_probs, original_decimal_odds, original_stake, corr=corr
    )
    p_rest = (
        correlation_adjusted_joint_prob(remaining_probs, corr)
        if corr is not None and len(remaining_probs) > 1
        else float(np.prod(remaining_probs)) if remaining_probs else 1.0
    )
    # utility: slightly discount the risky ride
    ride_utility = fair - risk_aversion * (fair - offered_cashout) * (1 - p_rest)
    recommend = offered_cashout >= ride_utility
    fair_odds = (1.0 / p_rest) if p_rest > 1e-9 else 999.0
    return CashoutDecision(
        ride_ev=fair,
        cashout_value=offered_cashout,
        cashout_ev=offered_cashout,
        recommend_cashout=recommend,
        edge_of_riding=fair - offered_cashout,
        remaining_combined_p=p_rest,
        fair_remaining_odds=fair_odds,
    )
