"""
Segmented CLV and calibration reports.

Segments: NFL | other_sports | PM | 1-leg | multi-leg | by book
Calibration: predicted model_p bins vs actual win rate.
"""

from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import numpy as np
from .history import BetHistory, _conn


def _infer_segment(row: dict) -> Dict[str, str]:
    src = (row.get("source") or "").lower()
    desc = (row.get("description") or "").upper()
    n_legs = int(row.get("n_legs") or 1)
    sport_col = (row.get("sport") or "").upper()

    if sport_col:
        sport = sport_col
    elif "NFL" in desc or src == "nfl":
        sport = "NFL"
    elif any(x in desc for x in ("NBA", "NHL", "MLB", "NCAAF", "EPL", "MLS")):
        for x in ("NBA", "NHL", "MLB", "NCAAF", "EPL", "MLS"):
            if x in desc:
                sport = x
                break
        else:
            sport = "OTHER"
    elif src in ("prediction", "polymarket", "kalshi") or desc.startswith("PM"):
        sport = "PM"
    else:
        sport = "OTHER"

    domain = "PM" if sport == "PM" else ("NFL" if sport == "NFL" else "OTHER_SPORTS")
    legs = "1-leg" if n_legs <= 1 else "multi"
    book = (row.get("book") or "unknown").lower() or "unknown"
    return {"sport": sport, "domain": domain, "legs": legs, "book": book}


@dataclass
class SegmentStats:
    key: str
    n: int
    pnl: float
    roi: float
    win_rate: float
    avg_clv: Optional[float]
    positive_clv_rate: Optional[float]
    avg_model_p: Optional[float]


def _stats_for_rows(rows: List[dict], key: str) -> SegmentStats:
    if not rows:
        return SegmentStats(key, 0, 0.0, 0.0, 0.0, None, None, None)
    pnls = [float(r["pnl"] or 0) for r in rows]
    stakes = [float(r["stake"] or 0) for r in rows]
    wins = sum(1 for r in rows if r.get("won") == 1)
    clvs = [float(r["clv"]) for r in rows if r.get("clv") is not None]
    ps = [float(r["model_p"]) for r in rows if r.get("model_p") is not None]
    st = sum(stakes) or 1.0
    return SegmentStats(
        key=key,
        n=len(rows),
        pnl=float(sum(pnls)),
        roi=float(sum(pnls) / st),
        win_rate=float(wins / len(rows)),
        avg_clv=float(np.mean(clvs)) if clvs else None,
        positive_clv_rate=float(np.mean([c > 0 for c in clvs])) if clvs else None,
        avg_model_p=float(np.mean(ps)) if ps else None,
    )


def segmented_clv_report(hist: BetHistory) -> Dict[str, List[SegmentStats]]:
    with _conn(hist.db_path) as c:
        rows = [dict(r) for r in c.execute(
            "SELECT * FROM bets WHERE status IN ('won','lost','cashed')"
        ).fetchall()]

    buckets: Dict[str, Dict[str, List[dict]]] = {
        "domain": defaultdict(list),
        "legs": defaultdict(list),
        "book": defaultdict(list),
        "sport": defaultdict(list),
    }
    for r in rows:
        seg = _infer_segment(r)
        buckets["domain"][seg["domain"]].append(r)
        buckets["legs"][seg["legs"]].append(r)
        buckets["book"][seg["book"]].append(r)
        buckets["sport"][seg["sport"]].append(r)

    out: Dict[str, List[SegmentStats]] = {}
    for axis, groups in buckets.items():
        out[axis] = [_stats_for_rows(v, k) for k, v in sorted(groups.items(), key=lambda x: -len(x[1]))]
    return out


def calibration_report(hist: BetHistory, n_bins: int = 6) -> List[dict]:
    """Reliability-style bins: mean predicted p vs actual win rate."""
    with _conn(hist.db_path) as c:
        rows = [dict(r) for r in c.execute(
            "SELECT model_p, won, stake, pnl FROM bets WHERE status IN ('won','lost') AND model_p IS NOT NULL"
        ).fetchall()]
    if not rows:
        return []

    edges = np.linspace(0.35, 0.75, n_bins + 1)
    report = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        bucket = [r for r in rows if lo <= float(r["model_p"]) < hi]
        if not bucket and i == len(edges) - 2:
            bucket = [r for r in rows if lo <= float(r["model_p"]) <= hi]
        if not bucket:
            report.append({
                "bin": f"{lo:.2f}-{hi:.2f}", "n": 0,
                "avg_pred": None, "actual": None, "gap": None,
            })
            continue
        avg_pred = float(np.mean([float(r["model_p"]) for r in bucket]))
        actual = float(np.mean([1.0 if r["won"] == 1 else 0.0 for r in bucket]))
        report.append({
            "bin": f"{lo:.2f}-{hi:.2f}",
            "n": len(bucket),
            "avg_pred": round(avg_pred, 3),
            "actual": round(actual, 3),
            "gap": round(actual - avg_pred, 3),  # negative = overconfident
        })
    return report


def edge_shrink_by_calibration(hist: BetHistory) -> float:
    """
    Global shrink factor from calibration.
    If model systematically overconfident (actual < pred), shrink edges.
    """
    cal = calibration_report(hist)
    usable = [b for b in cal if b["n"] >= 3 and b["gap"] is not None]
    if len(usable) < 2:
        return 1.0
    # weight by n
    gaps = np.array([b["gap"] for b in usable], dtype=float)
    weights = np.array([b["n"] for b in usable], dtype=float)
    mean_gap = float(np.average(gaps, weights=weights))
    # overconfident (negative gap) → shrink; underconfident → mild expand capped
    if mean_gap >= -0.01:
        return 1.0
    # e.g. gap -0.05 → shrink to 0.75
    shrink = float(np.clip(1.0 + 5.0 * mean_gap, 0.5, 1.0))
    return shrink


def print_analytics(hist: BetHistory) -> None:
    print("\n" + "─" * 72)
    print("  SEGMENTED CLV")
    print("─" * 72)
    report = segmented_clv_report(hist)
    for axis in ("domain", "legs", "sport", "book"):
        segs = report.get(axis) or []
        if not segs:
            continue
        print(f"\n  [{axis}]")
        for s in segs:
            clv = f"{s.avg_clv:+.4f}" if s.avg_clv is not None else "—"
            pos = f"{100*s.positive_clv_rate:.0f}%" if s.positive_clv_rate is not None else "—"
            print(f"    {s.key:<14} n={s.n:<3} ROI={100*s.roi:+.1f}%  win={100*s.win_rate:.0f}%  "
                  f"CLV={clv}  +CLV={pos}")

    print("\n" + "─" * 72)
    print("  CALIBRATION (pred p vs actual)")
    print("─" * 72)
    cal = calibration_report(hist)
    if not cal:
        print("  (need settled wins/losses with model_p)")
    else:
        for b in cal:
            if b["n"] == 0:
                continue
            gap = b["gap"]
            flag = " OVER" if gap is not None and gap < -0.03 else (" UNDER" if gap is not None and gap > 0.03 else "")
            print(f"    {b['bin']}: n={b['n']}  pred={b['avg_pred']:.3f}  actual={b['actual']:.3f}  gap={gap:+.3f}{flag}")
    shrink = edge_shrink_by_calibration(hist)
    print(f"\n  Calibration edge multiplier: ×{shrink:.2f}")
