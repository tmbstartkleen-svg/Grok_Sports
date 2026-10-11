#!/usr/bin/env python3
"""
Full-stack demo: live-style slate → ranked legs → correlation parlays →
portfolio → cash-out → 5-day rolling simulation.
Also demonstrates strict combined-p ≥ 0.52 mode.
"""

from __future__ import annotations
import os
import sys
from pathlib import Path

# Make imports work no matter where the project lives on disk
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
from core import (
    score_diff_monte_carlo, expected_value, edge, Leg, rank_legs,
    build_parlays, build_counterbalanced_portfolio, simulate_portfolio,
    estimate_same_game_correlation, correlation_adjusted_joint_prob,
    evaluate_cashout, RollingPortfolio,
)
from data.odds_connector import OddsConnector


def markets_to_legs(markets, seed=42):
    """Real consensus edge model (no random boost)."""
    from core.edge_model import markets_to_model_rows
    rows = markets_to_model_rows(markets, prefer_books=None)
    legs = []
    for i, r in enumerate(rows):
        model_p = float(r["model_p"])
        dec = float(r["best_decimal"])
        logit = np.log(max(model_p, 1e-6) / max(1 - model_p, 1e-6))
        mc = score_diff_monte_carlo(
            mean_margin=logit * 4.0,
            std_margin=11.0, n_sims=8_000, seed=seed + i,
        )
        p = float(np.clip(0.55 * mc.p + 0.45 * model_p, 0.12, 0.90))
        legs.append(Leg(
            id=f"{r['event_id']}|{r['selection'][:10]}",
            description=f"{r['sport']} {r['selection']} ({r['best_book']})",
            p=p, decimal_odds=dec,
            american_odds=float(r["best_american"]),
            edge=float(edge(p, dec)),
            ev=float(expected_value(p, dec)),
            ci_low=mc.ci_low, ci_high=mc.ci_high,
            sport=r["sport"],
            start_time=r["commence"].strftime("%m-%d") if hasattr(r["commence"], "strftime") else "?",
            correlation_group=r["event_id"],
        ))
    return legs


def main():
    print("=" * 72)
    print("SPORTS EV ENGINE – FULL STACK DEMO")
    print("=" * 72)

    connector = OddsConnector()
    try:
        markets = connector.fetch_slate(days_ahead=5, seed=7)
        source = "LIVE (The Odds API)" if connector.api_key else "synthetic"
    except Exception as e:
        print(f"\n[Data] Live fetch failed ({e}); using synthetic slate.")
        # Force synthetic path
        markets = connector._synthetic_slate(
            ["NFL", "NBA", "NHL", "MLB", "NCAAF"], days_ahead=5, seed=7
        )
        source = "synthetic (fallback)"
    print(f"\n[Data] {source}: {len(markets)} markets across ~5 days")

    legs = markets_to_legs(markets, seed=7)
    ranked = rank_legs(legs, min_edge=0.008, min_p=0.48)[:30]
    print(f"[Legs] {len(ranked)} high-EV legs after filters (showing top 8)")
    for l in ranked[:8]:
        print(f"  {l.id[:22]:<22} p={l.p:.3f} edge={l.edge:.3f} EV={l.ev:.4f} {l.start_time}")

    # Correlation demo on first 4
    if len(ranked) >= 4:
        ids = [l.id for l in ranked[:4]]
        types = ["h2h"] * 4
        corr = estimate_same_game_correlation(ids, types)
        joint = correlation_adjusted_joint_prob([l.p for l in ranked[:4]], corr, n_sims=20_000)
        indep = float(np.prod([l.p for l in ranked[:4]]))
        print(f"\n[Corr] 4-leg joint p: independence={indep:.4f}  copula={joint:.4f}")

    # Growth parlays
    growth = build_parlays(ranked, max_legs=3, min_combined_p=0.28, min_ev=0.02, max_parlays=25)
    print(f"\n[Parlays growth] {len(growth)} (min comb p 0.28)")
    for p in growth[:5]:
        print(f"  {p.n_legs}leg combP={p.combined_p:.3f} odds={p.decimal_odds:.2f} EV={p.ev:.4f}")

    # Strict mode: high-p legs only so 2-leg can clear 0.52
    strict_legs = rank_legs(legs, min_edge=0.005, min_p=0.58)[:40]
    strict = build_parlays(strict_legs, max_legs=2, min_combined_p=0.52, min_ev=0.005, max_parlays=20)
    print(f"\n[Parlays STRICT p≥0.52] {len(strict)} found (from {len(strict_legs)} high-p legs)")
    for p in strict[:5]:
        print(f"  {p.n_legs}leg combP={p.combined_p:.3f} odds={p.decimal_odds:.2f} EV={p.ev:.4f}")

    bankroll = 10_000.0
    positions = build_counterbalanced_portfolio(growth, bankroll, max_positions=5, fraction=0.25)
    print(f"\n[Portfolio] {len(positions)} positions, risk "
          f"${sum(p.stake for p in positions):.0f}")
    if positions:
        res = simulate_portfolio(positions, bankroll, n_sims=20_000, seed=11)
        print(f"  EV=${res.portfolio_ev:.0f}  P(profit)={100*res.prob_profit:.1f}%  "
              f"P(BE+)={100*res.prob_breakeven_or_better:.1f}%  ruin={100*res.ruin_prob:.2f}%")

    # Cash-out
    if growth:
        top = growth[0]
        offer = 50 * top.decimal_odds * top.combined_p * 0.88
        dec = evaluate_cashout([l.p for l in top.legs], top.decimal_odds, 50.0, offer)
        print(f"\n[Cash-out] offer=${dec.cashout_value:.2f} ride_ev=${dec.ride_ev:.2f} "
              f"→ {'CASH' if dec.recommend_cashout else 'RIDE'}")

    # Rolling 5-day
    print("\n[Rolling 5-day portfolio]")
    rp = RollingPortfolio(starting_bankroll=bankroll, max_risk_pct=0.12,
                          kelly_fraction=0.25, use_strict=False)
    for day in range(1, 6):
        day_parlays = build_parlays(ranked, max_legs=3, min_combined_p=0.28,
                                    min_ev=0.025, max_parlays=20)
        snap = rp.ingest_day(day, ranked, day_parlays)
        s = snap.portfolio_sim_summary
        cashed = sum(1 for r in snap.cashout_recommendations if r["action"] == "CASH_OUT")
        print(f"  Day {day}: bank=${snap.bankroll:,.0f} open={s['n_open']} "
              f"risk=${s['capital_at_risk']:.0f} EV=${s['portfolio_ev']:.0f} "
              f"P(profit)={100*s['prob_profit']:.0f}% cashed={cashed}")

    print("\n" + "=" * 72)
    print("All modules exercised. Launch dashboard with:")
    print("  streamlit run app.py")
    print("=" * 72)


if __name__ == "__main__":
    main()
