"""
Action queue: turn ranked legs/combos into a short, bettable list.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Any
from .kelly import kelly_stake
from .pm_combos import ComboTicket


@dataclass
class ActionItem:
    rank: int
    kind: str              # "leg" | "combo"
    title: str
    detail: str
    model_p: float
    decimal_odds: float
    edge: float
    ev: float
    stake: float
    book: str
    source: str
    n_legs: int
    ticket_id: str
    why: str
    legs: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "#": self.rank,
            "kind": self.kind,
            "title": self.title,
            "p": round(self.model_p, 3),
            "odds": round(self.decimal_odds, 2),
            "edge": round(self.edge, 3),
            "EV": round(self.ev, 4),
            "stake_$": round(self.stake, 2),
            "book": self.book,
            "source": self.source,
            "legs": self.n_legs,
            "why": self.why,
            "id": self.ticket_id,
        }


def build_action_queue(
    combos: Sequence[ComboTicket],
    bankroll: float,
    kelly_frac: float = 0.25,
    max_per_bet: float = 0.04,
    max_items: int = 8,
    min_edge: float = 0.01,
    min_ev: float = 0.015,
) -> List[ActionItem]:
    """Top actionable tickets only — empty queue is a valid output."""
    items: List[ActionItem] = []
    seen = set()
    ranked = sorted(combos, key=lambda t: t.ev, reverse=True)
    for t in ranked:
        if len(items) >= max_items:
            break
        if t.ev < min_ev or (t.n_legs == 1 and t.legs and t.legs[0].edge < min_edge):
            continue
        tid = t.id
        if tid in seen:
            continue
        # book from first leg description if present
        book = ""
        if t.legs:
            desc = t.legs[0].description
            for b in ("draftkings", "fanduel", "pinnacle", "betmgm", "polymarket", "kalshi"):
                if b in desc.lower():
                    book = b
                    break
            if not book and t.source == "prediction":
                book = "polymarket"
        stake = kelly_stake(bankroll, t.combined_p, t.decimal_odds, kelly_frac, max_pct=max_per_bet)
        if stake < 1:
            continue
        why_parts = [f"EV {t.ev:.3f}", f"p={t.combined_p:.3f}"]
        if t.n_legs == 1 and t.legs:
            why_parts.append(f"edge {t.legs[0].edge:.3f}")
        why_parts.append(f"{t.n_legs}leg")
        title = " + ".join(
            (l.description[:40] if hasattr(l, "description") else str(l)) for l in t.legs
        )
        items.append(ActionItem(
            rank=len(items) + 1,
            kind="leg" if t.n_legs == 1 else "combo",
            title=title[:100],
            detail=t.source,
            model_p=t.combined_p,
            decimal_odds=t.decimal_odds,
            edge=t.legs[0].edge if t.n_legs == 1 and t.legs else t.ev,
            ev=t.ev,
            stake=stake,
            book=book or t.source,
            source=t.source,
            n_legs=t.n_legs,
            ticket_id=tid,
            why=" · ".join(why_parts),
            legs=[{
                "id": l.id,
                "p": l.p,
                "odds": l.decimal_odds,
                "desc": l.description[:80],
            } for l in t.legs],
        ))
        seen.add(tid)
    return items
