#!/usr/bin/env python3
"""Grok Sports CLI desk. See README for commands."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import BetHistory, clv_gate, build_action_queue
from core.risk_limits import filter_actions_by_risk
from sports_network.storage.store import NetworkStore
from sports_network.orchestrator import run_all

BANNER = "GROK SPORTS desk — network pull | scan | q | report | history | open"

def _build_board(bankroll=10_000.0, kelly=0.25):
    import numpy as np
    from core import (
        score_diff_monte_carlo, expected_value, edge, Leg, rank_legs,
        markets_to_model_rows, legs_from_pred_markets, build_combo_tickets,
    )
    from data.prediction_markets import PredictionMarketConnector
    from sports_network.bridge import fetch_slate_with_network

    hist = BetHistory()
    gate = clv_gate(hist, base_kelly=kelly)
    legs, notes = [], []
    SEED, MIN_EDGE, MIN_P = 42, 0.015, 0.45

    markets, net_notes, injury_deltas = fetch_slate_with_network(
        sports=["NFL", "NCAAF", "NBA", "NHL"], days_ahead=5, seed=SEED, prefer_network=True
    )
    notes.extend(net_notes)
    rows = markets_to_model_rows(
        markets, prefer_books=None, apply_weather=True, apply_hfa=True,
        injury_deltas=injury_deltas or None,
    )
    for i, r in enumerate(rows):
        p = float(r["model_p"]); dec = float(r["best_decimal"])
        logit = np.log(max(p, 1e-6) / max(1 - p, 1e-6))
        mc = score_diff_monte_carlo(mean_margin=logit * 4.0, std_margin=11.0, n_sims=2500, seed=SEED + i)
        pp = float(np.clip(0.55 * mc.p + 0.45 * p, 0.12, 0.90))
        e = edge(pp, dec)
        if e < MIN_EDGE or pp < MIN_P:
            continue
        legs.append(Leg(
            id=f"{r['event_id']}|{r['selection'][:14]}",
            description=f"{r['sport']} {r['away']}@{r['home']} · {r['selection']} · {r['best_book']}",
            p=pp, decimal_odds=dec, american_odds=float(r["best_american"]),
            edge=float(e), ev=float(expected_value(pp, dec)),
            ci_low=mc.ci_low, ci_high=mc.ci_high, sport=r["sport"],
            start_time="?", correlation_group=r["event_id"],
        ))
    try:
        pms = PredictionMarketConnector(use_live=True).fetch_markets(
            limit=40, min_volume=200, seed=SEED, sources=["polymarket"]
        )
        notes.append(f"pm:{len(pms)}")
        if pms:
            legs.extend(legs_from_pred_markets(pms, min_edge=MIN_EDGE, min_p=max(0.28, MIN_P - 0.1)))
    except Exception:
        notes.append("pm:skip")

    ranked = rank_legs(legs, min_edge=MIN_EDGE, min_p=MIN_P)[:40]
    combos = build_combo_tickets(ranked, max_legs=3, min_combined_p=0.20, min_ev=0.015, max_tickets=40)
    if gate.edge_multiplier < 1.0:
        kept = []
        for t in combos:
            t.ev = t.ev * gate.edge_multiplier
            if t.ev >= 0.01:
                kept.append(t)
        combos = kept
    actions = build_action_queue(combos, bankroll, gate.kelly_fraction, 0.04, max_items=8, min_edge=MIN_EDGE * 0.8)
    try:
        actions, _ = filter_actions_by_risk(actions, hist, bankroll, 0.03, 0.15)
    except Exception:
        pass
    return {"notes": " · ".join(notes), "actions": actions, "n_legs": len(ranked),
            "n_combos": len(combos), "gate": gate, "hist": hist}

class Desk:
    def __init__(self):
        self.bankroll = 10_000.0
        self.kelly = 0.25
        self.last_board = None
        self.store = NetworkStore()

    def cmd_network_pull(self, _a=""):
        print("  Running bot team…")
        results = run_all(self.store)
        ok = sum(1 for v in results.values() if v.get("ok"))
        print(f"  Done · {ok}/{len(results)} bots ok")
        for name, r in results.items():
            st = "✓" if r.get("ok") else "✗"
            extra = f"n={r.get('n', 0)}" if r.get("ok") else str(r.get("error", ""))[:60]
            print(f"    {st} {name:22} {extra}")

    def cmd_scan(self, _a=""):
        print("  Scanning…")
        self.last_board = _build_board(self.bankroll, self.kelly)
        b = self.last_board
        print(f"  {b['notes']} · {b['n_legs']} legs · {b['n_combos']} combos")
        g = b["gate"]
        print(f"  Gate: {'OPEN' if g.allowed_to_scale else 'CLOSED'} · kelly={g.kelly_fraction:.2f} · {g.reason}")
        self.cmd_queue()

    def cmd_queue(self, _a=""):
        if not self.last_board:
            print("  No board — run scan first."); return
        actions = self.last_board["actions"]
        if not actions:
            print("  Empty action queue. Stand down."); return
        print(f"  Action queue ({len(actions)}):")
        for a in actions:
            print(f"  #{a.rank} {a.source[:10]:10} {a.n_legs}leg ${a.stake:.0f} p={a.model_p:.3f} "
                  f"odds={a.decimal_odds:.2f} edge={a.edge:.3f} {a.title[:48]}")

    def cmd_report(self, _a=""):
        hist = BetHistory(); gate = clv_gate(hist, base_kelly=self.kelly); s = hist.stats()
        print("  CLV / bankroll report")
        print(f"  Bankroll ${self.bankroll:,.0f} · settled {s['n']} · gate "
              f"{'OPEN' if gate.allowed_to_scale else 'CLOSED'} kelly={gate.kelly_fraction:.2f}")
        if s["n"]:
            print(f"  PnL ${s['pnl']:+.0f} ROI {100*s['roi']:.1f}% win {100*s['win_rate']:.0f}%")
        avg = s.get("avg_clv")
        print(f"  Avg CLV {avg:+.4f}" if avg is not None else "  Avg CLV —")
        print(f"  {gate.reason}")

    def cmd_history(self, args=""):
        n = 15
        try:
            if args.strip(): n = int(args.strip().split()[0])
        except Exception: pass
        rows = BetHistory().list_recent(n)
        if not rows: print("  No bets."); return
        for r in rows:
            clv = f"{r['clv']:+.4f}" if r.get("clv") is not None else "—"
            print(f"  #{r['id']} {r.get('status')} ${r.get('stake',0):.0f} clv={clv} {(r.get('description') or '')[:50]}")

    def cmd_open(self, _a=""):
        rows = BetHistory().list_open()
        if not rows: print("  No open positions."); return
        for r in rows:
            print(f"  #{r['id']} ${r.get('stake',0):.0f} {(r.get('description') or '')[:60]}")

    def dispatch(self, line):
        line = line.strip()
        if not line: return True
        parts = line.split(maxsplit=1)
        cmd, args = parts[0].lower(), (parts[1] if len(parts)>1 else "")
        if cmd in ("quit","exit","q!"): return False
        if cmd in ("help","?"): print(__doc__ or BANNER)
        elif (cmd=="network" and args.lower().startswith("pull")) or cmd=="collect": self.cmd_network_pull()
        elif cmd=="scan": self.cmd_scan()
        elif cmd in ("q","queue"): self.cmd_queue()
        elif cmd=="report": self.cmd_report()
        elif cmd=="history": self.cmd_history(args)
        elif cmd=="open": self.cmd_open()
        elif cmd=="bankroll":
            try: self.bankroll=float(args.strip()); print(f"  Bankroll ${self.bankroll:,.0f}")
            except Exception: print(f"  Bankroll ${self.bankroll:,.0f}")
        else: print(f"  Unknown: {cmd}")
        return True

    def run(self):
        print(BANNER)
        while True:
            try: line = input("grok-sports> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n  Bye."); break
            if not self.dispatch(line):
                print("  Bye."); break

def main():
    Desk().run()

if __name__ == "__main__":
    main()
