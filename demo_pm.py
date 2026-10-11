#!/usr/bin/env python3
"""Prediction markets + multi-leg path cash-out demo."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
from data.prediction_markets import PredictionMarketConnector
from data.odds_connector import OddsConnector
from core import (
    markets_to_model_rows, Leg, edge, expected_value, score_diff_monte_carlo,
    legs_from_pred_markets, build_combo_tickets, monte_carlo_combo_paths,
    kelly_stake, build_counterbalanced_portfolio, simulate_portfolio,
)
from core.pm_combos import ComboTicket
from core.portfolio import Position

def sports_legs(seed=7):
    c = OddsConnector(api_key=None)
    markets = c.fetch_slate(days_ahead=4, seed=seed)
    rows = markets_to_model_rows(markets, apply_weather=False)
    legs = []
    for i, r in enumerate(rows):
        p = float(r["model_p"])
        dec = float(r["best_decimal"])
        e = edge(p, dec)
        if e < 0.012 or p < 0.45:
            continue
        legs.append(Leg(
            id=f"{r['event_id']}|{r['selection'][:12]}",
            description=f"{r['sport']} {r['selection']}",
            p=p, decimal_odds=dec, american_odds=float(r["best_american"]),
            edge=float(e), ev=float(expected_value(p, dec)),
            ci_low=p-0.04, ci_high=p+0.04, sport=r["sport"],
            start_time="d", correlation_group=r["event_id"],
        ))
    return legs[:20]

def main():
    print("=" * 72)
    print("PREDICTION MARKETS + 1/2/3-LEG COMBOS + PATH CASH-OUT")
    print("=" * 72)

    pm = PredictionMarketConnector(use_live=True)
    pms = pm.fetch_markets(limit=40, min_volume=500, seed=3)
    pm_legs = legs_from_pred_markets(pms, min_edge=0.008, min_p=0.30)
    print(f"\n[PM] {len(pms)} outcomes → {len(pm_legs)} edged legs")
    for l in pm_legs[:5]:
        print(f"  {l.description[:62]} p={l.p:.3f} odds={l.decimal_odds:.2f} edge={l.edge:.3f}")

    s_legs = sports_legs()
    print(f"[Sports] {len(s_legs)} edged legs")

    all_legs = pm_legs + s_legs
    tickets = build_combo_tickets(
        all_legs, max_legs=3, min_combined_p=0.18, min_ev=0.02, max_tickets=35
    )
    print(f"\n[Combos] {len(tickets)} tickets (1–3 legs)")
    for t in tickets[:8]:
        print(f"  {t.n_legs}leg [{t.source:10}] p={t.combined_p:.3f} odds={t.decimal_odds:.2f} EV={t.ev:.4f}")
        print(f"         {t.id[:70]}")

    # Path cash-out MC on top multi-leg
    multi = [t for t in tickets if t.n_legs >= 2]
    if multi:
        top = multi[0]
        stake = 50.0
        stats = monte_carlo_combo_paths(top, stake, n_sims=8000, seed=11)
        print(f"\n[Path cash-out sim] {top.n_legs}-leg stake ${stake:.0f}")
        print(f"  mean PnL=${stats['mean_pnl']:.2f}  P(profit)={100*stats['p_profit']:.1f}%")
        print(f"  P(cash mid-path)={100*stats['p_cashout']:.1f}%  P(full win)={100*stats['p_full_win']:.1f}%  P(loss)={100*stats['p_loss']:.1f}%")

    # Portfolio of combos as positions
    bankroll = 10_000.0
    # adapt ComboTicket → pseudo Parlay-like positions
    positions = []
    used = set()
    for t in tickets:
        if len(positions) >= 6:
            break
        ids = {l.id for l in t.legs}
        if ids & used and t.n_legs > 1:
            continue
        st = kelly_stake(bankroll, t.combined_p, t.decimal_odds, fraction=0.25, max_pct=0.04)
        if st < 5:
            continue
        positions.append(Position(
            kind="combo", item=t, stake=st, p=t.combined_p, decimal_odds=t.decimal_odds,
        ))
        used |= ids
    if positions:
        # minimal sim
        res = simulate_portfolio(positions, bankroll, n_sims=12_000, seed=5)
        print(f"\n[Combo portfolio] {len(positions)} tickets risk ${sum(p.stake for p in positions):.0f}")
        print(f"  EV=${res.portfolio_ev:.0f}  P(profit)={100*res.prob_profit:.1f}%  P(BE+)={100*res.prob_breakeven_or_better:.1f}%")

    print("\n" + "=" * 72)
    print("Done. Mixed sports + prediction market board is live.")
    print("=" * 72)

if __name__ == "__main__":
    main()
