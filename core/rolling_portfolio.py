"""
Multi-day (1–7 day) rolling portfolio manager.

Keeps open positions, refreshes probabilities daily, suggests new entries,
hedges, and cash-outs as the slate evolves.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Sequence
import numpy as np

from .ev import Leg, expected_value
from .parlay import Parlay, build_parlays
from .portfolio import Position, simulate_portfolio, build_counterbalanced_portfolio
from .cashout import evaluate_cashout
from .kelly import fractional_kelly


@dataclass
class OpenPosition:
    pos_id: str
    kind: str                   # "leg" | "parlay"
    description: str
    stake: float
    decimal_odds: float
    combined_p_at_entry: float
    leg_ids: List[str]
    commence_times: List[datetime]
    entry_day: int
    status: str = "open"        # open | cashed | settled_win | settled_loss
    current_p: float = 0.0
    cashout_offer: float = 0.0


@dataclass
class DailySnapshot:
    day: int
    open_positions: List[OpenPosition]
    new_entries: List[Position]
    cashout_recommendations: List[Dict]
    portfolio_sim_summary: Dict
    bankroll: float


class RollingPortfolio:
    def __init__(
        self,
        starting_bankroll: float = 10_000.0,
        max_horizon_days: int = 7,
        max_risk_pct: float = 0.12,
        kelly_fraction: float = 0.25,
        strict_combined_p: float = 0.52,
        use_strict: bool = False,
    ):
        self.bankroll = starting_bankroll
        self.starting = starting_bankroll
        self.max_horizon = max_horizon_days
        self.max_risk_pct = max_risk_pct
        self.kelly_fraction = kelly_fraction
        self.strict_p = strict_combined_p
        self.use_strict = use_strict
        self.open: List[OpenPosition] = []
        self.history: List[DailySnapshot] = []
        self._pos_counter = 0

    def _new_id(self) -> str:
        self._pos_counter += 1
        return f"P{self._pos_counter:04d}"

    def ingest_day(
        self,
        day: int,
        ranked_legs: List[Leg],
        available_parlays: List[Parlay],
        updated_probs: Optional[Dict[str, float]] = None,
    ) -> DailySnapshot:
        """
        One daily cycle:
          1. Refresh probs / cash-out offers on open positions
          2. Decide cash-outs
          3. Add new counter-balanced entries under remaining risk budget
          4. Monte Carlo the resulting book
        """
        updated_probs = updated_probs or {}
        cashout_recs = []

        # 1–2. Refresh & cash-out logic
        still_open = []
        for op in self.open:
            if op.status != "open":
                continue
            # update combined p if we have leg-level updates
            if op.leg_ids and all(lid in updated_probs for lid in op.leg_ids):
                op.current_p = float(np.prod([updated_probs[lid] for lid in op.leg_ids]))
            else:
                op.current_p = op.combined_p_at_entry * np.random.uniform(0.92, 1.05)
                op.current_p = float(np.clip(op.current_p, 0.05, 0.95))

            # synthetic cash-out offer = fair * (0.85–0.95) haircut
            fair = op.stake * op.decimal_odds * op.current_p
            offer = fair * np.random.uniform(0.86, 0.94)
            op.cashout_offer = float(offer)

            decision = evaluate_cashout(
                remaining_probs=[op.current_p],  # simplified single remaining factor
                original_decimal_odds=op.decimal_odds,
                original_stake=op.stake,
                offered_cashout=offer,
                risk_aversion=0.15,
            )
            if decision.recommend_cashout and offer > op.stake * 1.05:
                op.status = "cashed"
                self.bankroll += offer
                cashout_recs.append({
                    "id": op.pos_id,
                    "action": "CASH_OUT",
                    "offer": offer,
                    "ride_ev": decision.ride_ev,
                    "edge": decision.edge_of_riding,
                })
            else:
                still_open.append(op)
                cashout_recs.append({
                    "id": op.pos_id,
                    "action": "RIDE",
                    "offer": offer,
                    "ride_ev": decision.ride_ev,
                    "edge": decision.edge_of_riding,
                })
        self.open = still_open

        # 3. New entries under residual risk budget
        capital_at_risk = sum(o.stake for o in self.open if o.status == "open")
        residual_budget = max(0.0, self.bankroll * self.max_risk_pct - capital_at_risk)

        # filter parlays by strictness
        min_p = self.strict_p if self.use_strict else 0.28
        candidates = [p for p in available_parlays if p.combined_p >= min_p and p.ev > 0.02]
        candidates = sorted(candidates, key=lambda x: x.ev, reverse=True)

        new_positions: List[Position] = []
        used_legs = set()
        for parl in candidates:
            if residual_budget < 5:
                break
            lids = {l.id for l in parl.legs}
            if lids & used_legs:
                continue
            f = fractional_kelly(parl.combined_p, parl.decimal_odds, self.kelly_fraction)
            stake = min(self.bankroll * f, residual_budget * 0.35)
            if stake < 5:
                continue
            pos = Position(
                kind="parlay",
                item=parl,
                stake=stake,
                p=parl.combined_p,
                decimal_odds=parl.decimal_odds,
            )
            new_positions.append(pos)
            op = OpenPosition(
                pos_id=self._new_id(),
                kind="parlay",
                description=parl.description[:60],
                stake=stake,
                decimal_odds=parl.decimal_odds,
                combined_p_at_entry=parl.combined_p,
                leg_ids=[l.id for l in parl.legs],
                commence_times=[],  # filled by caller if available
                entry_day=day,
                current_p=parl.combined_p,
            )
            self.open.append(op)
            used_legs |= lids
            residual_budget -= stake

        # 4. Simulate current book
        sim_positions = [
            Position(kind=o.kind, item=None, stake=o.stake, p=o.current_p or o.combined_p_at_entry,
                     decimal_odds=o.decimal_odds)
            for o in self.open if o.status == "open"
        ]
        if sim_positions:
            result = simulate_portfolio(sim_positions, self.bankroll, n_sims=12_000, seed=day + 7)
            summary = {
                "portfolio_ev": result.portfolio_ev,
                "mean_final": result.mean_final,
                "prob_profit": result.prob_profit,
                "prob_be": result.prob_breakeven_or_better,
                "ruin_prob": result.ruin_prob,
                "n_open": len(sim_positions),
                "capital_at_risk": sum(p.stake for p in sim_positions),
            }
        else:
            summary = {
                "portfolio_ev": 0.0, "mean_final": self.bankroll,
                "prob_profit": 0.0, "prob_be": 1.0, "ruin_prob": 0.0,
                "n_open": 0, "capital_at_risk": 0.0,
            }

        snap = DailySnapshot(
            day=day,
            open_positions=list(self.open),
            new_entries=new_positions,
            cashout_recommendations=cashout_recs,
            portfolio_sim_summary=summary,
            bankroll=self.bankroll,
        )
        self.history.append(snap)
        return snap
