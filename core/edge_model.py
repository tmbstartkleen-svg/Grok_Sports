"""
Real edge model for live odds.

Strategy (no random boost):
1. Group lines by (event_id, market, selection).
2. Devig multi-outcome markets when possible; else use multiplicative 2-way.
3. Consensus fair_p = median implied across books (or mean of middle 50%).
4. Soft edge = best available price vs consensus fair (line shopping).
5. Optional sharp prior: if pinnacle/circa present, blend toward sharp de-vigged p.
6. Model p = blend(consensus, sharp) with slight regression to 0.5 for uncertainty.
"""

from __future__ import annotations
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import numpy as np

from data.odds_connector import MarketOdds


SHARP_BOOKS = {"pinnacle", "circa", "bookmaker", "betfair_ex_uk", "betfair_ex_eu"}
SOFT_PREFERRED = {"draftkings", "fanduel", "betmgm", "caesars", "pointsbetus", "williamhill_us"}


def _devig_two_way(p1: float, p2: float) -> Tuple[float, float]:
    s = p1 + p2
    if s <= 0:
        return 0.5, 0.5
    return p1 / s, p2 / s


def build_consensus_and_best(
    markets: List[MarketOdds],
) -> Dict[Tuple[str, str, str], dict]:
    """
    Returns map (event_id, market, selection) -> {
        consensus_p, best_decimal, best_american, best_book,
        sharp_p (optional), n_books, home, away, sport, commence, selection
    }
    """
    groups: Dict[Tuple[str, str, str], List[MarketOdds]] = defaultdict(list)
    for m in markets:
        key = (m.event_id, m.market, m.selection)
        groups[key].append(m)

    # Also group by event+market for 2-way devig
    by_event_market: Dict[Tuple[str, str], List[MarketOdds]] = defaultdict(list)
    for m in markets:
        by_event_market[(m.event_id, m.market)].append(m)

    out = {}
    for key, lines in groups.items():
        event_id, market, selection = key
        implieds = [1.0 / m.decimal_odds for m in lines if m.decimal_odds > 1.0]
        if not implieds:
            continue
        implieds_sorted = sorted(implieds)
        # trimmed mean
        n = len(implieds_sorted)
        lo, hi = n // 4, max(n // 4 + 1, n - n // 4)
        consensus_raw = float(np.mean(implieds_sorted[lo:hi])) if hi > lo else float(np.median(implieds_sorted))

        # Best price for the bettor (highest decimal)
        best = max(lines, key=lambda m: m.decimal_odds)
        best_dec = best.decimal_odds
        best_am = best.american_odds
        best_book = best.book

        # Sharp prior if available
        sharp_lines = [m for m in lines if m.book.lower() in SHARP_BOOKS]
        sharp_p = None
        if sharp_lines:
            sharp_p = float(np.mean([1.0 / m.decimal_odds for m in sharp_lines]))

        # Try 2-way devig on same event+market if exactly 2 unique selections
        siblings = by_event_market.get((event_id, market), [])
        sel_set = {s.selection for s in siblings}
        if len(sel_set) == 2:
            # median implied per selection then devig
            sel_imp = {}
            for sel in sel_set:
                imps = [1.0 / s.decimal_odds for s in siblings if s.selection == sel and s.decimal_odds > 1]
                if imps:
                    sel_imp[sel] = float(np.median(imps))
            if selection in sel_imp and len(sel_imp) == 2:
                other = [v for k, v in sel_imp.items() if k != selection][0]
                fair_this, _ = _devig_two_way(sel_imp[selection], other)
                consensus_raw = fair_this

        # Model probability: blend consensus + sharp; light shrink only
        if sharp_p is not None:
            model_p = 0.40 * consensus_raw + 0.60 * sharp_p
        else:
            model_p = consensus_raw
        model_p = 0.92 * model_p + 0.08 * 0.5
        model_p = float(np.clip(model_p, 0.08, 0.92))

        # NFL Elo + spread prior + rest/injury blend
        sport = (best.sport or "").upper()
        if sport == "NFL":
            try:
                from core.nfl_model import get_nfl_model
                spread_home = None
                raw = getattr(best, "raw", None) or {}
                if isinstance(raw, dict) and raw.get("spread_home") is not None:
                    spread_home = float(raw["spread_home"])
                nfl_p = get_nfl_model().model_p_for_selection(
                    best.home, best.away, selection, spread_home=spread_home
                )
                if nfl_p is not None:
                    model_p = float(np.clip(0.40 * model_p + 0.60 * nfl_p, 0.08, 0.92))
            except Exception:
                pass


        # Line-shopping edge: model_p vs best price
        # If best decimal > fair decimal from model, positive EV
        out[key] = {
            "consensus_p": consensus_raw,
            "model_p": model_p,
            "best_decimal": best_dec,
            "best_american": best_am,
            "best_book": best_book,
            "sharp_p": sharp_p,
            "n_books": n,
            "home": best.home,
            "away": best.away,
            "sport": best.sport,
            "commence": best.commence_time,
            "selection": selection,
            "event_id": event_id,
            "market": market,
            "line": best.line,
        }
    return out


def markets_to_model_rows(
    markets: List[MarketOdds],
    prefer_books: Optional[List[str]] = None,
    injury_deltas: Optional[Dict[str, float]] = None,
    apply_weather: bool = True,
    apply_hfa: bool = True,
) -> List[dict]:
    """
    Collapse multi-book lines into one row per selection with real edge + factors.
    """
    from core.factors import (
        home_away_adjustment, weather_adjustment, injury_adjustment,
        combine_adjustments, apply_factors_to_p,
    )

    consensus = build_consensus_and_best(markets)
    rows = []
    for key, info in consensus.items():
        model_p = info["model_p"]
        dec = info["best_decimal"]
        book = info["best_book"]
        event_id, market, selection = key

        if prefer_books:
            candidates = [
                m for m in markets
                if m.event_id == event_id and m.market == market
                and m.selection == selection
                and m.book.lower() in {b.lower() for b in prefer_books}
            ]
            if not candidates:
                continue
            best_pref = max(candidates, key=lambda m: m.decimal_odds)
            dec = best_pref.decimal_odds
            book = best_pref.book

        adjs = []
        if apply_hfa:
            adjs.append(home_away_adjustment(
                selection, info.get("home", ""), info.get("away", ""),
                info.get("sport", ""),
            ))
        is_total = (
            "over" in selection.lower()
            or "under" in selection.lower()
            or market == "totals"
        )
        if apply_weather:
            adjs.append(weather_adjustment(
                info.get("sport", ""), info.get("home", ""),
                is_total_market=is_total,
            ))
        if injury_deltas:
            adjs.append(injury_adjustment(selection, injury_deltas))
        combined = combine_adjustments(*adjs) if adjs else None
        notes = []
        if combined:
            model_p = apply_factors_to_p(model_p, combined)
            notes = combined.notes
            factor_delta = combined.delta_p
        else:
            factor_delta = 0.0

        implied = 1.0 / dec if dec > 1 else 1.0
        edge_val = model_p - implied
        ev = model_p * (dec - 1.0) - (1.0 - model_p)
        rows.append({
            **info,
            "model_p": model_p,
            "best_decimal": dec,
            "best_book": book,
            "implied": implied,
            "edge": edge_val,
            "ev": ev,
            "factor_notes": notes,
            "factor_delta": factor_delta,
        })
    return rows
