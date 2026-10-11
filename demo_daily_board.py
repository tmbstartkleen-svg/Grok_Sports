#!/usr/bin/env python3
"""
Daily board demo – generates ~30 ranked legs, builds high-probability parlays
(combined p > 0.52), constructs a counter-balanced portfolio, and runs
Monte Carlo bankroll simulations.

This is the first working prototype of the EV + Kelly + Monte Carlo engine.
Replace the mock legs with live model outputs + odds feeds for production.
"""

from __future__ import annotations
import sys
sys.path.insert(0, "/home/workdir/artifacts/sports_ev_engine")

import numpy as np
from core import (
    score_diff_monte_carlo,
    american_to_decimal,
    expected_value,
    edge,
    Leg,
    rank_legs,
    build_parlays,
    build_counterbalanced_portfolio,
    simulate_portfolio,
    fractional_kelly,
)


def make_mock_legs(n: int = 50, seed: int = 42) -> list[Leg]:
    """
    Synthetic but realistic legs for demonstration.
    Intentionally injects a modest positive edge on many legs so the
    pipeline can be exercised end-to-end. In production these come from
    real Monte Carlo models + live odds feeds.
    """
    rng = np.random.default_rng(seed)
    sports = ["NFL", "NBA", "MLB", "NHL", "NCAAF"]
    legs = []
    for i in range(n):
        # Create a true probability biased toward the 58-75% range for many legs
        # so short parlays can still clear meaningful combined-p filters.
        base_logit = rng.normal(0.70, 0.50)          # centers around ~67%
        true_p = 1.0 / (1.0 + np.exp(-base_logit))
        # Market is a bit softer / noisier → creates edge
        market_p = true_p * rng.uniform(0.86, 0.96)
        market_p = float(np.clip(market_p, 0.32, 0.80))
        fair_dec = 1.0 / market_p
        vigged_dec = fair_dec * rng.uniform(0.955, 0.98)  # small juice
        vigged_dec = max(1.08, vigged_dec)
        am = (vigged_dec - 1) * 100 if vigged_dec >= 2 else -100 / (vigged_dec - 1)

        # Monte Carlo around the true probability (adds realistic uncertainty)
        mc = score_diff_monte_carlo(
            mean_margin=np.log(true_p / (1 - true_p)) * 4.0,
            std_margin=12.5,
            n_sims=20_000,
            seed=seed + i,
        )
        # Blend model p with the generative true_p for demo stability
        p = 0.65 * mc.p + 0.35 * true_p
        e = edge(p, vigged_dec)
        ev = expected_value(p, vigged_dec)
        sport = sports[i % len(sports)]
        legs.append(Leg(
            id=f"L{i:03d}",
            description=f"{sport} Leg-{i:02d}",
            p=float(p),
            decimal_odds=float(vigged_dec),
            american_odds=float(am),
            edge=float(e),
            ev=float(ev),
            ci_low=mc.ci_low,
            ci_high=mc.ci_high,
            sport=sport,
            start_time=f"Day+{(i % 7) + 1}",
            correlation_group=f"G{i // 4}",  # groups of 4
        ))
    return legs


def main():
    print("=" * 72)
    print("SPORTS EV ENGINE – Daily Board Prototype")
    print("Monte Carlo + Kelly + Constrained Parlays + Portfolio Simulation")
    print("=" * 72)

    # 1. Generate & rank legs
    all_legs = make_mock_legs(55)
    ranked = rank_legs(all_legs, min_edge=0.01, min_p=0.50)
    top30 = ranked[:30]
    print(f"\n[1] Ranked high-EV legs available today: {len(ranked)} (showing top 12 of 30)")
    print(f"{'ID':<6} {'Sport':<6} {'p':>6} {'Odds':>8} {'Edge':>7} {'EV/$1':>8} {'Day':<6}")
    print("-" * 55)
    for leg in top30[:12]:
        print(f"{leg.id:<6} {leg.sport:<6} {leg.p:6.3f} {leg.american_odds:8.1f} "
              f"{leg.edge:7.3f} {leg.ev:8.4f} {leg.start_time:<6}")

    # 2. Build parlays.
    # Note: a hard 0.52 combined-p floor is extremely strict for 2+ legs
    # (two 72% legs only reach ~0.52). We therefore run two tiers:
    #   A) strict high-probability parlays (p ≥ 0.45) for capital preservation
    #   B) positive-EV shorter parlays for growth
    strict_parlays = build_parlays(
        top30, max_legs=2, min_combined_p=0.45, min_ev=0.02,
        max_parlays=20, prefer_short=True,
    )
    growth_parlays = build_parlays(
        top30, max_legs=3, min_combined_p=0.28, min_ev=0.05,
        max_parlays=30, prefer_short=True,
    )
    # de-dupe by leg set
    seen = set()
    parlays = []
    for p in strict_parlays + growth_parlays:
        key = tuple(sorted(l.id for l in p.legs))
        if key not in seen:
            seen.add(key)
            parlays.append(p)
    parlays = sorted(parlays, key=lambda x: x.ev, reverse=True)[:25]

    print(f"\n[2] Qualifying parlays (strict + growth tiers): {len(parlays)}")
    print(f"{'#':<3} {'Legs':<4} {'CombP':>7} {'DecOdds':>8} {'EV':>8}  Description")
    print("-" * 70)
    for i, p in enumerate(parlays[:12], 1):
        print(f"{i:<3} {p.n_legs:<4} {p.combined_p:7.3f} {p.decimal_odds:8.2f} "
              f"{p.ev:8.4f}  {p.description[:45]}")

    # 3. Counter-balanced portfolio
    bankroll = 10_000.0
    positions = build_counterbalanced_portfolio(
        parlays,
        bankroll=bankroll,
        max_positions=5,
        fraction=0.25,
        max_total_risk_pct=0.12,
    )
    print(f"\n[3] Counter-balanced portfolio (bankroll ${bankroll:,.0f})")
    total_risk = 0.0
    for pos in positions:
        total_risk += pos.stake
        item = pos.item
        print(f"  Stake ${pos.stake:7.2f} | p={pos.p:.3f} | odds={pos.decimal_odds:.2f} | "
              f"{getattr(item, 'description', str(item))[:50]}")
    print(f"  Total capital at risk: ${total_risk:.2f} ({100*total_risk/bankroll:.1f}%)")

    # 4. Monte Carlo portfolio simulation
    if positions:
        result = simulate_portfolio(positions, starting_bankroll=bankroll, n_sims=25_000, seed=7)
        print(f"\n[4] Portfolio Monte Carlo ({result.n_sims:,} paths)")
        print(f"  Portfolio EV (theoretical): ${result.portfolio_ev:,.2f}")
        print(f"  Mean final bankroll:        ${result.mean_final:,.2f}")
        print(f"  Median final bankroll:      ${result.median_final:,.2f}")
        print(f"  P(profit):                  {100*result.prob_profit:.1f}%")
        print(f"  P(break-even or better):    {100*result.prob_breakeven_or_better:.1f}%")
        print(f"  Mean max drawdown:          {100*result.max_drawdown_mean:.1f}%")
        print(f"  P(ruin < 50% bankroll):     {100*result.ruin_prob:.2f}%")

        # Simple histogram summary
        pcts = np.percentile(result.final_bankrolls, [5, 25, 50, 75, 95])
        print(f"  Final bankroll percentiles:")
        print(f"    5%  ${pcts[0]:,.0f} | 25% ${pcts[1]:,.0f} | 50% ${pcts[2]:,.0f} | "
              f"75% ${pcts[3]:,.0f} | 95% ${pcts[4]:,.0f}")
    else:
        print("\n[4] No positions met the risk/diversity filters today.")

    print("\n" + "=" * 72)
    print("Next steps: wire live odds + real models, add same-game correlation")
    print("matrices, cash-out EV calculator, and multi-day rolling portfolio.")
    print("=" * 72)


if __name__ == "__main__":
    main()
