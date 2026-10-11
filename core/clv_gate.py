"""
CLV-gated Kelly: don't raise aggressiveness until rolling CLV is positive.

Policy:
- base_kelly from user (e.g. 0.25)
- if settled bets with CLV < min_n → use conservative_kelly (e.g. 0.10)
- if rolling avg CLV <= 0 → shrink kelly further
- if rolling avg CLV > 0 and positive_clv_rate >= threshold → allow full base_kelly
- optional edge_shrink multiplier when calibration is poor
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any
from .history import BetHistory
from .analytics import edge_shrink_by_calibration


@dataclass
class GateDecision:
    kelly_fraction: float
    edge_multiplier: float
    reason: str
    avg_clv: Optional[float]
    n_clv: int
    positive_clv_rate: Optional[float]
    allowed_to_scale: bool


def clv_gate(
    hist: BetHistory,
    base_kelly: float = 0.25,
    conservative_kelly: float = 0.10,
    min_clv_samples: int = 5,
    min_positive_rate: float = 0.52,
) -> GateDecision:
    s = hist.stats()
    n = int(s.get("clv_n") or 0)
    avg = s.get("avg_clv")
    pos_rate = s.get("positive_clv_rate")

    # Not enough evidence → stay small
    cal_mult = edge_shrink_by_calibration(hist)
    if n < min_clv_samples:
        return GateDecision(
            kelly_fraction=min(base_kelly, conservative_kelly),
            edge_multiplier=0.75 * cal_mult,
            reason=f"insufficient CLV samples ({n}/{min_clv_samples}) — conservative Kelly",
            avg_clv=avg,
            n_clv=n,
            positive_clv_rate=pos_rate,
            allowed_to_scale=False,
        )

    # Negative or flat CLV → shrink hard
    if avg is None or avg <= 0:
        return GateDecision(
            kelly_fraction=min(base_kelly, conservative_kelly * 0.8),
            edge_multiplier=0.5,
            reason=f"rolling CLV {avg} ≤ 0 — edge shrunk, Kelly capped",
            avg_clv=avg,
            n_clv=n,
            positive_clv_rate=pos_rate,
            allowed_to_scale=False,
        )

    # Positive CLV but weak hit rate
    if pos_rate is not None and pos_rate < min_positive_rate:
        return GateDecision(
            kelly_fraction=min(base_kelly, 0.15),
            edge_multiplier=0.7,
            reason=f"CLV+ but only {100*pos_rate:.0f}% positive — partial scale",
            avg_clv=avg,
            n_clv=n,
            positive_clv_rate=pos_rate,
            allowed_to_scale=False,
        )

    # Healthy track record — still apply calibration shrink
    cal_mult = edge_shrink_by_calibration(hist)
    return GateDecision(
        kelly_fraction=base_kelly,
        edge_multiplier=float(cal_mult),
        reason=f"CLV gate OPEN — avg CLV {avg:+.4f}, {100*(pos_rate or 0):.0f}% positive (n={n}), cal×{cal_mult:.2f}",
        avg_clv=avg,
        n_clv=n,
        positive_clv_rate=pos_rate,
        allowed_to_scale=True,
    )


def apply_edge_gate(raw_edge: float, gate: GateDecision, min_edge: float = 0.01) -> float:
    """Shrink displayed/usable edge; may drop below min → filter out."""
    e = raw_edge * gate.edge_multiplier
    return e if e >= min_edge else 0.0
