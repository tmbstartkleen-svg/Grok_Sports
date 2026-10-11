#!/usr/bin/env python3
"""
EV Desk — terminal UI (no Streamlit).
Action queue · log bets · open positions · path cash-out · CLV history
"""

from __future__ import annotations
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import json
import numpy as np

from core import (
    score_diff_monte_carlo, expected_value, edge, Leg, rank_legs,
    kelly_stake, markets_to_model_rows,
    legs_from_pred_markets, build_combo_tickets, monte_carlo_combo_paths,
    BetHistory, BetRecord, build_action_queue, evaluate_cashout,
    clv_gate, apply_edge_gate,
    print_analytics, check_risk, filter_actions_by_risk,
)
from core.portfolio import Position, build_counterbalanced_portfolio, simulate_portfolio
from data.odds_connector import OddsConnector
from data.prediction_markets import PredictionMarketConnector
try:
    from sports_network.client import NetworkClient
    from sports_network.bridge import apply_network_to_models, network_feature_bundle
    from sports_network.orchestrator import run_all
except Exception:
    NetworkClient = None
    apply_network_to_models = None
    network_feature_bundle = None
    run_all = None

# ── defaults ────────────────────────────────────────────────────────────
BANKROLL = float(os.environ.get("BANKROLL", "10000"))
KELLY = float(os.environ.get("KELLY", "0.25"))
MAX_PER = float(os.environ.get("MAX_PER_BET", "0.04"))
MIN_EDGE = float(os.environ.get("MIN_EDGE", "0.015"))
MIN_P = float(os.environ.get("MIN_P", "0.45"))
DAYS = int(os.environ.get("DAYS", "5"))
SEED = int(os.environ.get("SEED", "42"))
DAILY_LOSS = float(os.environ.get("DAILY_LOSS_PCT", "0.03"))
SPORT_CAP = float(os.environ.get("SPORT_CAP_PCT", "0.15"))


def hr(title: str = ""):
    print("\n" + "─" * 72)
    if title:
        print(f"  {title}")
        print("─" * 72)


def scan(use_sports=True, use_pm=True, live_sports=True, live_pm=True, kelly_frac=None, edge_mult=1.0):
    legs = []
    notes = []

    if use_sports:
        conn = OddsConnector() if live_sports else OddsConnector(api_key=None)
        try:
            markets = conn.fetch_slate(
                sports=["NFL", "NCAAF", "NBA", "NHL"],
                days_ahead=DAYS, seed=SEED,
            )
            notes.append(f"sports:{'LIVE' if conn.api_key else 'synth'}:{len(markets)}")
        except Exception as e:
            markets = OddsConnector(api_key=None)._synthetic_slate(
                ["NFL", "NCAAF", "NBA", "NHL"], DAYS, SEED
            )
            notes.append(f"sports:fallback")
        rows = markets_to_model_rows(markets, prefer_books=None, apply_weather=True, apply_hfa=True)
        for i, r in enumerate(rows):
            p = float(r["model_p"])
            dec = float(r["best_decimal"])
            logit = np.log(max(p, 1e-6) / max(1 - p, 1e-6))
            mc = score_diff_monte_carlo(
                mean_margin=logit * 4.0, std_margin=11.0, n_sims=3000, seed=SEED + i
            )
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
                start_time=r["commence"].strftime("%Y-%m-%d") if hasattr(r["commence"], "strftime") else "?",
                correlation_group=r["event_id"],
            ))

    if use_pm:
        pmc = PredictionMarketConnector(use_live=live_pm)
        try:
            pms = pmc.fetch_markets(limit=40, min_volume=200, seed=SEED, sources=["polymarket"])
        except Exception:
            pms = pmc.fetch_markets(limit=30, seed=SEED, sources=[])
            pms = pmc._synthetic(30, SEED)
        notes.append(f"pm:{len(pms)}")
        legs.extend(legs_from_pred_markets(pms, min_edge=MIN_EDGE, min_p=max(0.28, MIN_P - 0.1)))

    ranked = rank_legs(legs, min_edge=MIN_EDGE, min_p=MIN_P)[:40]
    combos = build_combo_tickets(
        ranked, max_legs=3, min_combined_p=0.20, min_ev=0.015, max_tickets=40
    )
    kf = kelly_frac if kelly_frac is not None else KELLY
    # Apply edge gate: drop combos whose edge shrinks below floor
    if edge_mult < 1.0:
        filtered = []
        for t in combos:
            adj_ev = t.ev * edge_mult
            if adj_ev >= 0.01:
                t.ev = adj_ev
                filtered.append(t)
        combos = filtered
    actions = build_action_queue(
        combos, BANKROLL, kf, MAX_PER, max_items=8, min_edge=MIN_EDGE * (1.0 if edge_mult >= 1 else 0.8)
    )
    # risk filter applied in main after hist is available — placeholder
    risk_meta = {"daily_loss": DAILY_LOSS, "sport_cap": SPORT_CAP}

    positions = []
    counts = {}
    for t in combos:
        if len(positions) >= 6:
            break
        ids = [l.id for l in t.legs]
        if any(counts.get(i, 0) >= 2 for i in ids):
            continue
        st_ = kelly_stake(BANKROLL, t.combined_p, t.decimal_odds, kf, max_pct=MAX_PER)
        if st_ < 5:
            continue
        positions.append(Position(
            kind="combo", item=t, stake=st_, p=t.combined_p, decimal_odds=t.decimal_odds
        ))
        for i in ids:
            counts[i] = counts.get(i, 0) + 1

    result = None
    if positions:
        result = simulate_portfolio(positions, BANKROLL, n_sims=12_000, seed=SEED + 99)

    path = None
    multi = [t for t in combos if t.n_legs >= 2]
    if multi:
        top = multi[0]
        path = monte_carlo_combo_paths(top, 50.0, n_sims=5000, seed=SEED + 3)
        path["ticket"] = top

    return {
        "notes": " · ".join(notes),
        "ranked": ranked,
        "combos": combos,
        "actions": actions,
        "positions": positions,
        "result": result,
        "path": path,
    }


def show_clv(hist: BetHistory, gate=None):
    s = hist.stats()
    hr("CLV / HISTORY / GATE")
    print(f"  Settled: {s['n']}   ROI: {100*s['roi']:.1f}%   Win%: {100*s['win_rate']:.0f}%   PnL: ${s['pnl']:.2f}")
    if s["avg_clv"] is not None:
        print(f"  Avg CLV: {s['avg_clv']:+.4f}   Positive CLV rate: {100*(s['positive_clv_rate'] or 0):.0f}%  (n={s['clv_n']})")
    else:
        print("  Avg CLV: — (settle bets with close price to populate)")
    if s.get("avg_slippage") is not None:
        print(f"  Avg execution slippage (dec odds): {s['avg_slippage']:+.4f}  (<0 = worse than board)")
    if gate is not None:
        status = "OPEN" if gate.allowed_to_scale else "CLOSED"
        print(f"  Kelly gate: {status}  kelly={gate.kelly_fraction:.2f}  edge×{gate.edge_multiplier:.2f}")
        print(f"  → {gate.reason}")
    recent = hist.list_recent(10)
    if recent:
        print("\n  Recent:")
        for r in recent:
            clv = f"{r['clv']:+.4f}" if r["clv"] is not None else "—"
            pnl = f"{r['pnl']:+.1f}" if r["pnl"] is not None else "—"
            print(f"    #{r['id']:<4} {r['status']:<7} stake=${r['stake']:<7.0f} pnl={pnl:<8} clv={clv}  {(r['description'] or '')[:40]}")


def show_actions(actions, hist: BetHistory):
    hr("ACTION QUEUE")
    if not actions:
        print("  (empty) — no ticket cleared filters. Stand down.")
        return
    for a in actions:
        print(f"\n  #{a.rank}  [{a.source}]  {a.n_legs}leg  book={a.book or '—'}")
        print(f"      {a.title[:68]}")
        print(f"      stake ${a.stake:.0f}   p={a.model_p:.3f}   odds {a.decimal_odds:.2f}   edge {a.edge:.3f}   EV {a.ev:.4f}")
        print(f"      why: {a.why}")


def show_open(hist: BetHistory):
    hr("OPEN POSITIONS")
    open_bets = hist.list_open()
    if not open_bets:
        print("  None. Log a bet from the action queue (command: log N).")
        return
    for ob in open_bets:
        print(f"\n  #{ob['id']}  ${ob['stake']:.0f}  odds {ob['decimal_odds']:.2f}  p={ob['model_p']:.3f}")
        print(f"      {ob['description'][:68]}")
        remaining = []
        try:
            remaining = json.loads(ob["remaining_json"] or "[]")
        except Exception:
            pass
        if remaining:
            rem_ps = [float(x.get("p", 0.5)) for x in remaining]
            fair = float(ob["stake"]) * float(ob["decimal_odds"]) * float(np.prod(rem_ps))
            offer = fair * 0.93
            dec = evaluate_cashout(
                rem_ps, float(ob["decimal_odds"]), float(ob["stake"]), offer, risk_aversion=0.08
            )
            action = "CASH" if dec.recommend_cashout else "RIDE"
            print(f"      offer ${offer:.2f}  ride_ev ${dec.ride_ev:.2f}  → {action}")
            print(f"      commands: cash {ob['id']} | win {ob['id']} | lose {ob['id']}")


def show_path(path):
    hr("PATH CASH-OUT SIM (top multi-leg @ $50)")
    if not path:
        print("  No 2+ leg combo available.")
        return
    top = path["ticket"]
    print(f"  {top.n_legs}leg [{top.source}]  p={top.combined_p:.3f}  odds={top.decimal_odds:.2f}")
    print(f"  {top.id[:70]}")
    print(f"  mean PnL ${path['mean_pnl']:.2f}   P(profit) {100*path['p_profit']:.1f}%   "
          f"P(mid-cash) {100*path['p_cashout']:.1f}%   P(full win) {100*path['p_full_win']:.1f}%")


def show_portfolio(result, positions):
    hr("PORTFOLIO")
    if not positions or not result:
        print("  No positions sized.")
        return
    risk = sum(p.stake for p in positions)
    print(f"  {len(positions)} tickets   risk ${risk:.0f}")
    print(f"  EV ${result.portfolio_ev:.0f}   P(profit) {100*result.prob_profit:.1f}%   "
          f"P(BE+) {100*result.prob_breakeven_or_better:.1f}%   ruin {100*result.ruin_prob:.2f}%")


def help_text():
    hr("COMMANDS")
    print("""  scan              refresh sports + PM board
  scan offline      synthetic only (no APIs)
  q / queue          show action queue
  log N [odds]      log action #N (optional executed decimal odds)
  open              list open positions + cash/ride
  exec ID odds      set executed price on open bet
  cash ID           cash out open bet ID
  win ID            settle open bet as full win (stores CLV vs slight closer)
  lose ID           settle open bet as loss
  clv / history     CLV stats + recent bets
  report            segmented CLV + calibration
  risk              daily stop + exposure caps
  network           sports network summary
  network pull      run bot team collect
  network top       NFL ratings snapshot
  path              path cash-out simulation
  port              portfolio summary
  legs              top ranked legs
  help              this text
  quit / exit       leave desk
""")


def main():
    hist = BetHistory()
    board = None
    gate = clv_gate(hist, base_kelly=KELLY)

    print("=" * 72)
    print("  GROK SPORTS  ·  desk  ·  bankroll ${:,.0f}  ·  base kelly {:.0%}".format(BANKROLL, KELLY))
    print("=" * 72)
    print("  No API key required. network pull → scan. Type help.")
    print("  Type scan to build the board, then q for the action queue.")
    print("  help for commands. Empty queue = do not bet.")
    print("  NFL Elo model active on NFL moneylines. CLV gate controls Kelly.")

    show_clv(hist, gate)

    while True:
        try:
            raw = input("\ngrok-sports> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye.")
            break
        if not raw:
            continue
        parts = raw.split()
        cmd = parts[0].lower()
        args = parts[1:]

        if cmd in ("quit", "exit", "q") and not args:
            # bare q shows queue if board exists; "quit" exits
            if cmd == "q" and board is not None:
                show_actions(board["actions"], hist)
                continue
            if cmd in ("quit", "exit"):
                print("bye.")
                break
            if cmd == "q":
                print("  Run scan first.")
                continue

        if cmd == "quit" or cmd == "exit":
            print("bye.")
            break

        if cmd == "help":
            help_text()
            continue

        if cmd == "scan":
            offline = (args and args[0].lower() == "offline")
            print("  Scanning…")
            if apply_network_to_models:
                try:
                    applied = apply_network_to_models()
                    print(f"  Network applied: {applied}")
                except Exception as e:
                    print(f"  Network apply skip: {e}")
            gate = clv_gate(hist, base_kelly=KELLY)
            board = scan(
                use_sports=True, use_pm=True,
                live_sports=not offline,  # network/ESPN — no Odds API key required
                # (Odds API only if ODDS_API_KEY set inside odds_bot)
                live_pm=not offline,
                kelly_frac=gate.kelly_fraction,
                edge_mult=gate.edge_multiplier,
            )
            print(f"  Gate: kelly={gate.kelly_fraction:.2f} edge×{gate.edge_multiplier:.2f} — {gate.reason}")
            board["actions"], risk_st = filter_actions_by_risk(
                board["actions"], hist, BANKROLL, DAILY_LOSS, SPORT_CAP
            )
            print(f"  Risk: {risk_st.reason}  daily_pnl=${risk_st.daily_pnl:.0f}")
            if risk_st.sport_exposure:
                print(f"  Exposure: { {k: round(v,0) for k,v in risk_st.sport_exposure.items()} }")
            print(f"  {board['notes']}")
            print(f"  {len(board['ranked'])} legs · {len(board['combos'])} combos · {len(board['actions'])} actionable")
            show_actions(board["actions"], hist)
            show_portfolio(board["result"], board["positions"])
            show_path(board["path"])
            continue

        if cmd in ("queue",):
            if not board:
                print("  Run scan first.")
            else:
                show_actions(board["actions"], hist)
            continue

        if cmd == "log":
            if not board:
                print("  Run scan first.")
                continue
            if not args or not args[0].isdigit():
                print("  usage: log N [executed_decimal]   e.g. log 1 1.91")
                continue
            n = int(args[0])
            match = [a for a in board["actions"] if a.rank == n]
            if not match:
                print(f"  No action #{n}")
                continue
            a = match[0]
            sport = "NFL" if "NFL" in a.title.upper() else (
                "PM" if a.source in ("prediction", "polymarket", "kalshi") else "OTHER"
            )
            rst = check_risk(hist, BANKROLL, DAILY_LOSS, SPORT_CAP, a.stake, sport)
            if not rst.ok:
                print(f"  BLOCKED: {rst.reason}")
                continue
            executed = a.decimal_odds
            if len(args) >= 2:
                try:
                    executed = float(args[1])
                except ValueError:
                    print("  executed_decimal must be a number")
                    continue
            bid = hist.log_bet(
                BetRecord(
                    ticket_id=a.ticket_id,
                    description=a.title,
                    n_legs=a.n_legs,
                    source=a.source,
                    model_p=a.model_p,
                    decimal_odds=executed,
                    stake=a.stake,
                    book=a.book,
                    edge=a.edge,
                    notes=a.why,
                    raw={"legs": a.legs},
                    board_decimal=a.decimal_odds,
                    executed_decimal=executed,
                ),
                legs=a.legs,
            )
            slip = executed - a.decimal_odds
            print(f"  Logged open bet #{bid}  stake ${a.stake:.0f}")
            print(f"  board odds {a.decimal_odds:.3f}  executed {executed:.3f}  slippage {slip:+.3f}")
            continue

        if cmd == "open":
            show_open(hist)
            continue

        if cmd == "exec":
            if len(args) < 2:
                print("  usage: exec ID DECIMAL_ODDS   (price you actually got)")
                continue
            try:
                bid = int(args[0]); exe = float(args[1])
            except ValueError:
                print("  usage: exec ID DECIMAL_ODDS")
                continue
            hist.set_executed(bid, exe)
            print(f"  Bet #{bid} executed odds set to {exe:.3f}")
            continue

        if cmd == "cash":
            if not args or not args[0].isdigit():
                print("  usage: cash ID")
                continue
            bid = int(args[0])
            opens = {o["id"]: o for o in hist.list_open()}
            if bid not in opens:
                print(f"  No open bet #{bid}")
                continue
            ob = opens[bid]
            remaining = json.loads(ob["remaining_json"] or "[]")
            rem_ps = [float(x.get("p", 0.5)) for x in remaining] or [0.5]
            fair = float(ob["stake"]) * float(ob["decimal_odds"]) * float(np.prod(rem_ps))
            offer = fair * 0.93
            hist.cashout(bid, offer, float(ob["stake"]))
            print(f"  Cashed #{bid} @ ${offer:.2f}  pnl ${offer - float(ob['stake']):+.2f}")
            continue

        if cmd == "win":
            if not args or not args[0].isdigit():
                print("  usage: win ID")
                continue
            bid = int(args[0])
            opens = {o["id"]: o for o in hist.list_open()}
            if bid not in opens:
                print(f"  No open bet #{bid}")
                continue
            ob = opens[bid]
            stake = float(ob["stake"])
            odds = float(ob["decimal_odds"])
            pnl = stake * (odds - 1)
            # slight closer for demo CLV (replace with real close when you have it)
            close = odds * 0.98
            hist.settle(bid, True, pnl, close_decimal=close)
            print(f"  Won #{bid}  pnl ${pnl:+.2f}  close odds {close:.2f}")
            continue

        if cmd == "lose":
            if not args or not args[0].isdigit():
                print("  usage: lose ID")
                continue
            bid = int(args[0])
            opens = {o["id"]: o for o in hist.list_open()}
            if bid not in opens:
                print(f"  No open bet #{bid}")
                continue
            hist.mark_leg_lost(bid, float(opens[bid]["stake"]))
            print(f"  Lost #{bid}")
            continue

        if cmd in ("clv", "history"):
            gate = clv_gate(hist, base_kelly=KELLY)
            show_clv(hist, gate)
            continue

        if cmd in ("report", "analytics", "cal"):
            print_analytics(hist)
            continue

        if cmd == "network":
            if NetworkClient is None:
                print("  Sports network not available")
                continue
            sub = (args[0].lower() if args else "status")
            if sub == "pull":
                print("  Running bot team…")
                if run_all:
                    run_all()
                else:
                    NetworkClient().collect()
                print("  Done.")
                continue
            if sub == "top":
                c = NetworkClient()
                for team, elo in (network_feature_bundle(c).get("top_nfl") or [])[:12]:
                    print(f"    {team:<4} elo={elo}")
                continue
            c = NetworkClient()
            s = c.summary()
            print("  Inventory:")
            for row in s.get("inventory") or []:
                print(f"    {row.get('domain'):10} {row.get('sport'):6} n={row.get('n')}")
            print("  Recent runs:")
            for row in (s.get("recent_runs") or [])[:8]:
                print(f"    {row.get('bot_name'):20} {row.get('n_records')} @ {row.get('ran_at')}")
            continue

        if cmd == "risk":
            st = check_risk(hist, BANKROLL, DAILY_LOSS, SPORT_CAP)
            print(f"  {st.reason}")
            print(f"  daily_pnl=${st.daily_pnl:.2f}  limit=${st.daily_loss_limit:.2f}")
            print(f"  exposure={st.sport_exposure}  blocked={st.blocked_sports}")
            continue

        if cmd == "path":
            if not board:
                print("  Run scan first.")
            else:
                show_path(board["path"])
            continue

        if cmd in ("port", "portfolio"):
            if not board:
                print("  Run scan first.")
            else:
                show_portfolio(board["result"], board["positions"])
            continue

        if cmd == "legs":
            if not board:
                print("  Run scan first.")
                continue
            hr("TOP LEGS")
            for l in board["ranked"][:12]:
                src = "PM" if l.sport == "PRED" else l.sport
                print(f"  [{src}] p={l.p:.3f} edge={l.edge:.3f} odds={l.decimal_odds:.2f}  {l.description[:55]}")
            continue

        print(f"  Unknown command: {cmd}. Type help.")


if __name__ == "__main__":
    main()
