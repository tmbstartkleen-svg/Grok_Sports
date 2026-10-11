#!/usr/bin/env python3
"""Quick NFL Elo sanity backtest on synthetic closes."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from core.nfl_model import NFLModel, DEFAULT_RATINGS

def main():
    rng = np.random.default_rng(7)
    model = NFLModel()
    teams = list(DEFAULT_RATINGS.keys())
    bank = 10_000.0
    stake = 50.0
    n = 200
    wins = clv_pos = 0
    pnl = 0.0
    clvs = []

    print("=" * 60)
    print("NFL ELO BACKTEST (synthetic games, model vs juiced market)")
    print("=" * 60)

    for i in range(n):
        home, away = rng.choice(teams, size=2, replace=False)
        p_h, p_a = model.win_prob(home, away)
        # true outcome from model + noise
        true_p = float(np.clip(p_h + rng.normal(0, 0.03), 0.15, 0.85))
        home_won = rng.random() < true_p
        # market: model + juice noise
        mkt_p = float(np.clip(p_h + rng.uniform(-0.04, 0.04), 0.2, 0.8))
        mkt_dec = 1.0 / (mkt_p * 1.04)  # juice
        # bet home if model edge vs market
        edge = p_h - (1.0 / mkt_dec)
        if edge < 0.02:
            model.update_from_result(home, away, home_won)
            continue
        # close slightly efficient
        close_p = float(np.clip(true_p + rng.normal(0, 0.01), 0.2, 0.8))
        close_dec = 1.0 / close_p
        taken_imp = 1.0 / mkt_dec
        close_imp = 1.0 / close_dec
        clv = close_imp - taken_imp
        clvs.append(clv)
        if clv > 0:
            clv_pos += 1
        if home_won:
            wins += 1
            pnl += stake * (mkt_dec - 1)
        else:
            pnl -= stake
        model.update_from_result(home, away, home_won)

    print(f"Bets: {len(clvs)} / {n} games")
    print(f"Win rate: {100*wins/max(1,len(clvs)):.1f}%")
    print(f"PnL @ ${stake:.0f}/bet: ${pnl:.0f}")
    print(f"Avg CLV: {np.mean(clvs):+.4f}" if clvs else "Avg CLV: —")
    print(f"Positive CLV rate: {100*clv_pos/max(1,len(clvs)):.0f}%")
    print("=" * 60)
    print("This is a model-consistency check, not a claim of live profit.")

if __name__ == "__main__":
    main()
