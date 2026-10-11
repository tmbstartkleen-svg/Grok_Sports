"""
1 / 2 / 3 leg combo builder across sports + prediction markets.
Mid-path cash-out simulation: after k legs resolve, decide cash vs ride.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Sequence, Optional, Tuple
import itertools
import numpy as np

from .ev import Leg, expected_value, edge
from .kelly import fractional_kelly, kelly_stake
from .cashout import evaluate_cashout


@dataclass
class ComboTicket:
    legs: List[Leg]
    combined_p: float
    decimal_odds: float
    ev: float
    n_legs: int
    source: str = "mixed"   # sports | prediction | mixed

    @property
    def id(self) -> str:
        return " + ".join(l.id[:18] for l in self.legs)


def legs_from_pred_markets(markets, min_edge: float = 0.01, min_p: float = 0.35) -> List[Leg]:
    """Turn PredMarket rows into Leg objects with model p from fair/raw."""
    legs = []
    for m in markets:
        fair = float(m.raw.get("fair_p", m.implied_prob))
        # mild confidence shrink
        model_p = 0.9 * fair + 0.1 * 0.5
        model_p = float(np.clip(model_p, 0.08, 0.92))
        dec = float(m.decimal_odds)
        e = edge(model_p, dec)
        if e < min_edge or model_p < min_p:
            continue
        legs.append(Leg(
            id=m.market_id,
            description=f"PM {m.question[:50]} · {m.outcome}",
            p=model_p,
            decimal_odds=dec,
            american_odds=(dec - 1) * 100 if dec >= 2 else -100 / (dec - 1),
            edge=float(e),
            ev=float(expected_value(model_p, dec)),
            ci_low=max(0.05, model_p - 0.05),
            ci_high=min(0.95, model_p + 0.05),
            sport="PRED",
            start_time=m.end_date or "?",
            correlation_group=m.event_slug or m.market_id.split("|")[0],
        ))
    return legs


def build_combo_tickets(
    legs: Sequence[Leg],
    max_legs: int = 3,
    min_combined_p: float = 0.15,
    min_ev: float = 0.02,
    max_tickets: int = 40,
    allow_same_event: bool = False,
) -> List[ComboTicket]:
    """Enumerate 1..max_legs combinations with diversity constraint."""
    ranked = sorted(legs, key=lambda l: l.ev, reverse=True)[:36]
    tickets: List[ComboTicket] = []

    # 1-leg
    for l in ranked:
        if l.p >= min_combined_p and l.ev >= min_ev * 0.5:
            tickets.append(ComboTicket(
                legs=[l], combined_p=l.p, decimal_odds=l.decimal_odds,
                ev=l.ev, n_legs=1,
                source="prediction" if l.sport == "PRED" else "sports",
            ))

    for n in range(2, max_legs + 1):
        for combo in itertools.combinations(ranked, n):
            groups = [l.correlation_group for l in combo]
            if not allow_same_event and len(set(groups)) < n:
                continue  # skip same-event parlays unless allowed
            cp = float(np.prod([l.p for l in combo]))
            if cp < min_combined_p:
                continue
            dec = float(np.prod([l.decimal_odds for l in combo]))
            ev = expected_value(cp, dec)
            if ev < min_ev:
                continue
            src = "prediction" if all(l.sport == "PRED" for l in combo) else (
                "sports" if all(l.sport != "PRED" for l in combo) else "mixed"
            )
            tickets.append(ComboTicket(
                legs=list(combo), combined_p=cp, decimal_odds=dec,
                ev=float(ev), n_legs=n, source=src,
            ))

    tickets.sort(key=lambda t: t.ev, reverse=True)
    return tickets[:max_tickets]


@dataclass
class PathCashoutSim:
    """Simulate cash-out decisions as legs resolve one by one."""
    ticket: ComboTicket
    stake: float
    path_results: List[dict] = field(default_factory=list)
    final_pnl: float = 0.0
    action_sequence: List[str] = field(default_factory=list)


def simulate_midpath_cashout(
    ticket: ComboTicket,
    stake: float,
    leg_outcomes: Sequence[bool],
    offer_haircut: float = 0.93,
    risk_aversion: float = 0.08,
) -> PathCashoutSim:
    """
    After each resolved leg, book offers ~haircut * fair remaining value.
    Decide CASH vs RIDE with evaluate_cashout. If all win and never cash → full payout.
    """
    sim = PathCashoutSim(ticket=ticket, stake=stake)
    remaining = list(ticket.legs)
    outcomes = list(leg_outcomes)
    original_odds = ticket.decimal_odds

    for step in range(len(ticket.legs)):
        if not remaining:
            break
        # resolve current first remaining leg
        won = outcomes[step] if step < len(outcomes) else False
        if not won:
            sim.action_sequence.append(f"L{step+1}_LOSS")
            sim.final_pnl = -stake
            sim.path_results.append({"step": step + 1, "result": "LOSS", "pnl": -stake})
            return sim

        remaining = remaining[1:]
        if not remaining:
            # all legs won
            payout = stake * original_odds
            sim.action_sequence.append("WIN_FULL")
            sim.final_pnl = payout - stake
            sim.path_results.append({"step": step + 1, "result": "WIN", "pnl": sim.final_pnl})
            return sim

        # mid-path offer
        p_rest = float(np.prod([l.p for l in remaining]))
        fair = stake * original_odds * p_rest
        offer = fair * offer_haircut
        dec = evaluate_cashout(
            remaining_probs=[l.p for l in remaining],
            original_decimal_odds=original_odds,
            original_stake=stake,
            offered_cashout=offer,
            risk_aversion=risk_aversion,
        )
        if dec.recommend_cashout:
            sim.action_sequence.append(f"L{step+1}_CASH")
            sim.final_pnl = offer - stake
            sim.path_results.append({
                "step": step + 1, "result": "CASH", "offer": offer,
                "ride_ev": dec.ride_ev, "pnl": sim.final_pnl,
            })
            return sim
        sim.action_sequence.append(f"L{step+1}_RIDE")
        sim.path_results.append({
            "step": step + 1, "result": "RIDE", "offer": offer, "ride_ev": dec.ride_ev,
        })

    return sim


def monte_carlo_combo_paths(
    ticket: ComboTicket,
    stake: float,
    n_sims: int = 5000,
    offer_haircut: float = 0.93,
    seed: int = 0,
) -> dict:
    """Distribution of PnL if we follow optimal mid-path cash-out policy."""
    rng = np.random.default_rng(seed)
    pnls = []
    cash_count = 0
    full_win = 0
    loss = 0
    for i in range(n_sims):
        outcomes = [rng.random() < l.p for l in ticket.legs]
        sim = simulate_midpath_cashout(ticket, stake, outcomes, offer_haircut=offer_haircut)
        pnls.append(sim.final_pnl)
        if "CASH" in "".join(sim.action_sequence):
            cash_count += 1
        elif sim.final_pnl > 0:
            full_win += 1
        else:
            loss += 1
    arr = np.asarray(pnls)
    return {
        "mean_pnl": float(np.mean(arr)),
        "median_pnl": float(np.median(arr)),
        "p_profit": float(np.mean(arr > 0)),
        "p_cashout": cash_count / n_sims,
        "p_full_win": full_win / n_sims,
        "p_loss": loss / n_sims,
        "stake": stake,
    }
