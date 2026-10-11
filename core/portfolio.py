"""
Portfolio of parlays / legs with Monte Carlo bankroll path simulation
and simple counter-balancing logic.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Sequence, Optional
from .parlay import Parlay
from .ev import Leg
from .kelly import fractional_kelly, kelly_stake


@dataclass
class Position:
    kind: str               # "leg" or "parlay"
    item: Leg | Parlay
    stake: float
    p: float
    decimal_odds: float


@dataclass
class PortfolioResult:
    positions: List[Position]
    total_stake: float
    portfolio_ev: float
    n_sims: int
    final_bankrolls: np.ndarray
    mean_final: float
    median_final: float
    prob_profit: float
    prob_breakeven_or_better: float
    max_drawdown_mean: float
    ruin_prob: float          # ended below 50% of starting


def simulate_portfolio(
    positions: Sequence[Position],
    starting_bankroll: float,
    n_sims: int = 20_000,
    seed: Optional[int] = None,
    correlation: float = 0.10,
) -> PortfolioResult:
    """
    Monte Carlo the joint outcomes of a set of positions.
    Simple Gaussian copula-style correlation on the latent variables.
    """
    rng = np.random.default_rng(seed)
    n = len(positions)
    if n == 0:
        empty = np.full(n_sims, starting_bankroll)
        return PortfolioResult([], 0.0, 0.0, n_sims, empty, starting_bankroll,
                              starting_bankroll, 0.0, 1.0, 0.0, 0.0)

    # latent normals with constant correlation
    cov = np.full((n, n), correlation)
    np.fill_diagonal(cov, 1.0)
    latents = rng.multivariate_normal(np.zeros(n), cov, size=n_sims)

    # convert to Bernoulli wins via probit (Gaussian copula)
    from scipy.special import erfinv
    thresholds = np.array([
        np.sqrt(2) * erfinv(2 * min(max(pos.p, 1e-6), 1 - 1e-6) - 1)
        for pos in positions
    ])
    wins = latents < thresholds[None, :]   # shape (n_sims, n)

    bankrolls = np.full(n_sims, starting_bankroll, dtype=float)
    total_stake = sum(p.stake for p in positions)
    running_min = bankrolls.copy()

    for i, pos in enumerate(positions):
        payout = np.where(wins[:, i], pos.stake * pos.decimal_odds, 0.0)
        # stake is risked
        bankrolls = bankrolls - pos.stake + payout
        running_min = np.minimum(running_min, bankrolls)

    portfolio_ev = sum(pos.stake * (pos.p * (pos.decimal_odds - 1) - (1 - pos.p))
                       for pos in positions)
    mean_final = float(np.mean(bankrolls))
    median_final = float(np.median(bankrolls))
    prob_profit = float(np.mean(bankrolls > starting_bankroll))
    prob_be = float(np.mean(bankrolls >= starting_bankroll * 0.98))
    max_dd = float(np.mean((starting_bankroll - running_min) / starting_bankroll))
    ruin = float(np.mean(bankrolls < starting_bankroll * 0.5))

    return PortfolioResult(
        positions=list(positions),
        total_stake=total_stake,
        portfolio_ev=portfolio_ev,
        n_sims=n_sims,
        final_bankrolls=bankrolls,
        mean_final=mean_final,
        median_final=median_final,
        prob_profit=prob_profit,
        prob_breakeven_or_better=prob_be,
        max_drawdown_mean=max_dd,
        ruin_prob=ruin,
    )


def build_counterbalanced_portfolio(
    top_parlays: List[Parlay],
    bankroll: float,
    max_positions: int = 6,
    fraction: float = 0.25,
    max_total_risk_pct: float = 0.15,
    max_leg_appearances: int = 2,
    max_overlap_frac: float = 0.34,
    prefer_breakeven: bool = True,
) -> List[Position]:
    """
    Greedy diversified portfolio builder.

    Rules that protect break-even:
    - Each unique leg may appear in at most `max_leg_appearances` selected parlays
      (default 2) so one bad game cannot sink the whole book.
    - Parlay-to-parlay leg overlap capped at `max_overlap_frac`.
    - Stakes are fractional Kelly, further scaled down when prefer_breakeven
      so total risk stays conservative.
    - Candidates sorted by EV / sqrt(n_legs) to favor efficient short parlays.
    """
    from collections import Counter

    # Rank candidates: high EV, fewer legs preferred (less variance)
    scored = sorted(
        top_parlays,
        key=lambda p: (p.ev / max(1, p.n_legs ** 0.5), p.combined_p),
        reverse=True,
    )

    positions: List[Position] = []
    leg_count: Counter = Counter()
    remaining_risk = bankroll * max_total_risk_pct
    be_scale = 0.85 if prefer_breakeven else 1.0

    for parl in scored:
        if len(positions) >= max_positions:
            break
        leg_ids = [l.id for l in parl.legs]
        # Cap repeated legs
        if any(leg_count[i] >= max_leg_appearances for i in leg_ids):
            continue
        # Overlap with already selected parlays
        if positions:
            all_used = set()
            for pos in positions:
                item = pos.item
                if hasattr(item, "legs"):
                    all_used |= {l.id for l in item.legs}
            overlap = len(set(leg_ids) & all_used) / max(1, len(leg_ids))
            if overlap > max_overlap_frac:
                continue

        f = fractional_kelly(parl.combined_p, parl.decimal_odds, fraction) * be_scale
        stake = min(bankroll * f, remaining_risk * 0.35)
        if stake < 1.0:
            continue
        positions.append(Position(
            kind="parlay",
            item=parl,
            stake=stake,
            p=parl.combined_p,
            decimal_odds=parl.decimal_odds,
        ))
        for i in leg_ids:
            leg_count[i] += 1
        remaining_risk -= stake
        if remaining_risk <= 0:
            break
    return positions


def leg_usage_report(positions: Sequence[Position]) -> dict:
    """How many times each leg appears — for UI transparency."""
    from collections import Counter
    c: Counter = Counter()
    for pos in positions:
        item = pos.item
        if hasattr(item, "legs"):
            for l in item.legs:
                c[l.id] += 1
        elif hasattr(item, "id"):
            c[item.id] += 1
    return dict(c)
