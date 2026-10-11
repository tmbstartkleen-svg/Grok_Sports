"""
Daily loss limits and exposure caps.
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence
from .history import BetHistory, _conn


def _today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


@dataclass
class RiskStatus:
    ok: bool
    daily_pnl: float
    daily_loss_limit: float
    reason: str
    sport_exposure: Dict[str, float]
    blocked_sports: List[str]


def daily_pnl(hist: BetHistory, day: Optional[str] = None) -> float:
    day = day or _today_utc()
    with _conn(hist.db_path) as c:
        rows = c.execute(
            """SELECT pnl FROM bets
               WHERE status IN ('won','lost','cashed')
               AND (settled_at LIKE ? OR created_at LIKE ?)""",
            (f"{day}%", f"{day}%"),
        ).fetchall()
    return float(sum(float(r["pnl"] or 0) for r in rows))


def open_exposure_by_sport(hist: BetHistory) -> Dict[str, float]:
    with _conn(hist.db_path) as c:
        rows = c.execute(
            "SELECT stake, description, source, n_legs FROM bets WHERE status='open'"
        ).fetchall()
    exp: Dict[str, float] = {}
    for r in rows:
        desc = (r["description"] or "").upper()
        src = (r["source"] or "").lower()
        if "NFL" in desc:
            key = "NFL"
        elif src in ("prediction", "polymarket", "kalshi") or desc.startswith("PM"):
            key = "PM"
        else:
            key = "OTHER"
        exp[key] = exp.get(key, 0.0) + float(r["stake"] or 0)
    return exp


def check_risk(
    hist: BetHistory,
    bankroll: float,
    daily_loss_pct: float = 0.03,
    sport_cap_pct: float = 0.15,
    proposed_stake: float = 0.0,
    proposed_sport: str = "OTHER",
) -> RiskStatus:
    """
    daily_loss_pct: stop new risk after this fraction of bankroll lost today.
    sport_cap_pct: max open exposure in one sport bucket.
    """
    d_pnl = daily_pnl(hist)
    limit = -abs(bankroll * daily_loss_pct)
    exp = open_exposure_by_sport(hist)
    blocked = []
    for sport, amt in exp.items():
        if amt >= bankroll * sport_cap_pct:
            blocked.append(sport)

    if d_pnl <= limit:
        return RiskStatus(
            ok=False,
            daily_pnl=d_pnl,
            daily_loss_limit=limit,
            reason=f"DAILY STOP: pnl ${d_pnl:.0f} ≤ limit ${limit:.0f} ({100*daily_loss_pct:.0f}% of bankroll)",
            sport_exposure=exp,
            blocked_sports=blocked,
        )

    sport_key = "NFL" if proposed_sport.upper() == "NFL" else (
        "PM" if proposed_sport.upper() in ("PM", "PRED", "PREDICTION") else "OTHER"
    )
    new_exp = exp.get(sport_key, 0.0) + proposed_stake
    if proposed_stake > 0 and new_exp > bankroll * sport_cap_pct:
        return RiskStatus(
            ok=False,
            daily_pnl=d_pnl,
            daily_loss_limit=limit,
            reason=f"EXPOSURE CAP: {sport_key} would be ${new_exp:.0f} > {100*sport_cap_pct:.0f}% bankroll",
            sport_exposure=exp,
            blocked_sports=blocked + [sport_key],
        )

    return RiskStatus(
        ok=True,
        daily_pnl=d_pnl,
        daily_loss_limit=limit,
        reason="risk OK",
        sport_exposure=exp,
        blocked_sports=blocked,
    )


def filter_actions_by_risk(actions, hist: BetHistory, bankroll: float,
                           daily_loss_pct: float = 0.03, sport_cap_pct: float = 0.15):
    """Drop action-queue items that would breach limits."""
    status = check_risk(hist, bankroll, daily_loss_pct, sport_cap_pct)
    if not status.ok and "DAILY STOP" in status.reason:
        return [], status

    kept = []
    for a in actions:
        sport = "PM" if (a.source or "").lower() in ("prediction", "polymarket", "kalshi") else (
            "NFL" if "NFL" in (a.title or "").upper() else "OTHER"
        )
        st = check_risk(
            hist, bankroll, daily_loss_pct, sport_cap_pct,
            proposed_stake=a.stake, proposed_sport=sport,
        )
        if st.ok:
            kept.append(a)
    return kept, status
