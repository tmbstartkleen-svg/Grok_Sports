#!/usr/bin/env python3
"""Sports EV Engine — elite desk: action queue, CLV history, path cash-out."""

from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import streamlit as st

from core import (
    score_diff_monte_carlo, expected_value, edge, Leg, rank_legs,
    build_parlays, build_counterbalanced_portfolio, simulate_portfolio,
    evaluate_cashout, kelly_stake, markets_to_model_rows,
    legs_from_pred_markets, build_combo_tickets, monte_carlo_combo_paths,
    BetHistory, BetRecord, build_action_queue,
    clv_gate,
)
from core.portfolio import Position
from core.cashout import evaluate_cashout as eval_co
from data.odds_connector import OddsConnector
from data.prediction_markets import PredictionMarketConnector

# ── elite visual theme ──────────────────────────────────────────────────
st.set_page_config(page_title="Grok Sports", layout="wide", page_icon="◆")
st.markdown("""
<style>
  .stApp { background: #0b0e11; color: #e8eaed; }
  h1, h2, h3 { color: #f0f3f6 !important; letter-spacing: -0.02em; }
  [data-testid="stSidebar"] { background: #11151a; border-right: 1px solid #1e2530; }
  div[data-testid="stMetricValue"] { font-size: 1.4rem; color: #e8eaed; }
  .action-card {
    background: linear-gradient(145deg, #12171e 0%, #0e1218 100%);
    border: 1px solid #1e2a38;
    border-radius: 12px;
    padding: 1rem 1.15rem;
    margin-bottom: 0.65rem;
  }
  .action-card:hover { border-color: #2d6cdf; }
  .tag {
    display: inline-block; font-size: 0.7rem; padding: 0.15rem 0.45rem;
    border-radius: 4px; margin-right: 0.35rem; font-weight: 600;
  }
  .tag-go { background: #0d3d2a; color: #3dd68c; }
  .tag-src { background: #1a2332; color: #8b9cb3; }
  .tag-book { background: #2a1f14; color: #e8a54b; }
  .muted { color: #8b9cb3; font-size: 0.85rem; }
  .hero-num { font-size: 1.75rem; font-weight: 700; color: #3dd68c; }
  .danger { color: #f07178; }
  .ok { color: #3dd68c; }
</style>
""", unsafe_allow_html=True)

st.markdown("# Grok Sports")
st.caption("Action queue · CLV ledger · Path cash-out · Sports + Polymarket + Kalshi")

hist = BetHistory()
gate = clv_gate(hist, base_kelly=0.25)

with st.expander("How to use this desk", expanded=False):
    st.markdown("""
1. **Generate board** → only tickets that clear edge / EV filters appear in the **Action queue**.  
2. **Log bet** stores stake, odds, model p for later CLV.  
3. **Open positions** → resolve legs → system compares cash-out offer vs ride EV.  
4. **CLV panel** — positive average CLV means you beat the closing line (skill signal).  
5. Empty queue is normal. No edge → no bet.
""")

# ── sidebar ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Capital")
    bankroll = st.number_input("Bankroll $", 100.0, value=10_000.0, step=500.0)
    kelly_frac = st.slider("Kelly fraction", 0.1, 1.0, 0.25, 0.05)
    max_risk = st.slider("Max risk %", 5, 25, 12) / 100.0
    max_per_bet = st.slider("Max per bet %", 1, 10, 4) / 100.0
    max_leg_app = st.slider("Max leg reuse", 1, 3, 2)

    st.markdown("### Sources")
    use_sports = st.checkbox("Sports", True)
    use_pm = st.checkbox("Prediction markets", True)
    pm_sources = st.multiselect("PM", ["polymarket", "kalshi"], ["polymarket"])
    live_sports = st.checkbox("Live sports API", True)
    live_pm = st.checkbox("Live PM", True)

    st.markdown("### Filters")
    days = st.slider("Days ahead", 1, 7, 5)
    book_mode = st.selectbox("Book", ["Best price (any US)", "DraftKings only", "FanDuel only"])
    sports = st.multiselect("Leagues", ["NFL","NCAAF","NBA","NHL","MLB"], ["NFL","NCAAF","NBA","NHL"])
    min_edge = st.slider("Min edge", 0.0, 0.08, 0.015, 0.005)
    min_p = st.slider("Min p", 0.30, 0.70, 0.45, 0.01)
    max_combo = st.slider("Max combo legs", 1, 3, 3)
    min_cp = st.slider("Min combined p", 0.12, 0.55, 0.22, 0.01)
    n_sims = st.selectbox("MC paths", [5000, 15000, 25000], index=1)
    seed = st.number_input("Seed", value=42, step=1)
    run = st.button("Generate board", type="primary", use_container_width=True)


def prefer(mode):
    return {"DraftKings only": ["draftkings"], "FanDuel only": ["fanduel"]}.get(mode)


@st.cache_data(show_spinner="Scanning markets…", ttl=90)
def scan(seed, days, bankroll, kelly_frac, max_risk, max_per_bet, max_leg_app,
         use_sports, use_pm, pm_sources, live_sports, live_pm, sports, book_mode,
         min_edge, min_p, max_combo, min_cp, n_sims, edge_mult=1.0):
    legs = []
    notes = []
    if use_sports:
        conn = OddsConnector() if live_sports else OddsConnector(api_key=None)
        try:
            markets = conn.fetch_slate(sports=list(sports) or None, days_ahead=days, seed=int(seed))
            notes.append(f"sports:{'LIVE' if conn.api_key else 'synth'}:{len(markets)}")
        except Exception as e:
            markets = OddsConnector(api_key=None)._synthetic_slate(list(sports) or ["NFL"], days, int(seed))
            notes.append(f"sports:fb")
        rows = markets_to_model_rows(markets, prefer_books=prefer(book_mode), apply_weather=True, apply_hfa=True)
        for i, r in enumerate(rows):
            p = float(r["model_p"]); dec = float(r["best_decimal"])
            logit = np.log(max(p, 1e-6) / max(1 - p, 1e-6))
            mc = score_diff_monte_carlo(mean_margin=logit * 4.0, std_margin=11.0, n_sims=3500, seed=int(seed) + i)
            pp = float(np.clip(0.55 * mc.p + 0.45 * p, 0.12, 0.90))
            e = edge(pp, dec)
            if e < min_edge or pp < min_p:
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
        pms = pmc.fetch_markets(limit=40, min_volume=200, seed=int(seed), sources=list(pm_sources) or ["polymarket"])
        notes.append(f"pm:{len(pms)}")
        legs.extend(legs_from_pred_markets(pms, min_edge=min_edge, min_p=max(0.28, min_p - 0.1)))

    ranked = rank_legs(legs, min_edge=min_edge, min_p=min_p)[:40]
    combos = build_combo_tickets(ranked, max_legs=max_combo, min_combined_p=min_cp, min_ev=0.015, max_tickets=40)
    if edge_mult < 1.0:
        kept = []
        for t in combos:
            t.ev = t.ev * edge_mult
            if t.ev >= 0.01:
                kept.append(t)
        combos = kept
    actions = build_action_queue(combos, bankroll, kelly_frac, max_per_bet, max_items=8, min_edge=min_edge)


    # portfolio from combos with leg cap
    positions, used, counts = [], set(), {}
    for t in combos:
        if len(positions) >= 8:
            break
        ids = [l.id for l in t.legs]
        if any(counts.get(i, 0) >= max_leg_app for i in ids):
            continue
        st_ = kelly_stake(bankroll, t.combined_p, t.decimal_odds, kelly_frac, max_pct=max_per_bet)
        if st_ < 5 or sum(p.stake for p in positions) + st_ > bankroll * max_risk:
            continue
        positions.append(Position(kind="combo", item=t, stake=st_, p=t.combined_p, decimal_odds=t.decimal_odds))
        for i in ids:
            counts[i] = counts.get(i, 0) + 1
            used.add(i)
    result = simulate_portfolio(positions, bankroll, n_sims=n_sims, seed=int(seed) + 99) if positions else None

    path = None
    multi = [t for t in combos if t.n_legs >= 2]
    if multi:
        top = multi[0]
        path = monte_carlo_combo_paths(top, 50.0, n_sims=min(8000, n_sims), seed=int(seed) + 3)
        path["ticket"] = top

    return {
        "notes": " · ".join(notes),
        "ranked": ranked,
        "combos": combos,
        "actions": actions,
        "positions": positions,
        "result": result,
        "path": path,
        "usage": counts,
    }


# ── top metrics strip ───────────────────────────────────────────────────
stats = hist.stats()
m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("CLV bets", stats["clv_n"])
m2.metric("Avg CLV", f"{stats['avg_clv']:+.4f}" if stats["avg_clv"] is not None else "—")
m3.metric("Settled ROI", f"{100*stats['roi']:.1f}%" if stats["n"] else "—")
m4.metric("Win rate", f"{100*stats['win_rate']:.0f}%" if stats["n"] else "—")
m5.metric("Net PnL $", f"{stats['pnl']:.0f}" if stats["n"] else "—")
m6.metric("Kelly gate", "OPEN" if gate.allowed_to_scale else "CLOSED")
st.caption(f"Gate: kelly={gate.kelly_fraction:.2f} · edge×{gate.edge_multiplier:.2f} · {gate.reason}")

if run:
    gate = clv_gate(hist, base_kelly=kelly_frac)
    data = scan(
        int(seed), days, bankroll, gate.kelly_fraction, max_risk, max_per_bet, max_leg_app,
        use_sports, use_pm, tuple(pm_sources), live_sports, live_pm, tuple(sports),
        book_mode, min_edge, min_p, max_combo, min_cp, n_sims, gate.edge_multiplier,
    )
    st.session_state["board"] = data
else:
    data = st.session_state.get("board")

if not data:
    st.info("Configure sources → **Generate board**. Empty action queue means no bet — that’s correct.")
    st.stop()

res = data["result"]
st.markdown(f"<p class='muted'>{data['notes']} · {len(data['ranked'])} legs · {len(data['combos'])} combos</p>", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Actionable", len(data["actions"]))
c2.metric("Portfolio EV $", f"{res.portfolio_ev:.0f}" if res else "—")
c3.metric("P(break-even+)", f"{100*res.prob_breakeven_or_better:.1f}%" if res else "—")
c4.metric("Ruin", f"{100*res.ruin_prob:.2f}%" if res else "—")

# ── ACTION QUEUE (primary) ──────────────────────────────────────────────
st.markdown("## Action queue")
st.caption("Only positive-EV tickets after filters. Log a bet to track CLV.")

if not data["actions"]:
    st.success("No actionable tickets. Stand down — protect the bankroll.")
else:
    for a in data["actions"]:
        with st.container():
            st.markdown(f"""
<div class="action-card">
  <span class="tag tag-go">#{a.rank}</span>
  <span class="tag tag-src">{a.source}</span>
  <span class="tag tag-book">{a.book or '—'}</span>
  <span class="tag tag-src">{a.n_legs} leg</span>
  <div style="margin-top:0.5rem;font-weight:600">{a.title}</div>
  <div class="muted" style="margin-top:0.35rem">{a.why}</div>
  <div style="margin-top:0.6rem">
    <span class="hero-num">${a.stake:.0f}</span>
    <span class="muted"> stake · p={a.model_p:.3f} · odds {a.decimal_odds:.2f} · edge {a.edge:.3f}</span>
  </div>
</div>
""", unsafe_allow_html=True)
            b1, b2, b3 = st.columns([1, 1, 4])
            if b1.button("Log bet", key=f"log_{a.rank}", use_container_width=True):
                bid = hist.log_bet(
                    BetRecord(
                        ticket_id=a.ticket_id,
                        description=a.title,
                        n_legs=a.n_legs,
                        source=a.source,
                        model_p=a.model_p,
                        decimal_odds=a.decimal_odds,
                        stake=a.stake,
                        book=a.book,
                        edge=a.edge,
                        notes=a.why,
                        raw={"legs": a.legs},
                        board_decimal=a.decimal_odds,
                        executed_decimal=a.decimal_odds,
                    ),
                    legs=a.legs,
                )
                st.toast(f"Logged bet #{bid}", icon="✓")
                st.rerun()
            b2.code(f"{a.decimal_odds:.2f} @ {a.book}", language=None)

# ── tabs ────────────────────────────────────────────────────────────────
tab_pos, tab_path, tab_hist, tab_legs = st.tabs(
    ["Open positions", "Path cash-out", "CLV / history", "All legs"]
)

with tab_pos:
    open_bets = hist.list_open()
    if not open_bets:
        st.caption("No open positions. Log a bet from the action queue.")
    for ob in open_bets:
        st.markdown(f"**#{ob['id']}** · ${ob['stake']:.0f} · {ob['description'][:70]}")
        remaining = []
        try:
            import json as _json
            remaining = _json.loads(ob["remaining_json"] or "[]")
        except Exception:
            pass
        st.caption(f"p={ob['model_p']:.3f} · odds {ob['decimal_odds']:.2f} · resolved legs={ob.get('legs_resolved') or 0}")
        if remaining:
            # synthetic path decision for UI
            rem_ps = [float(x.get("p", 0.5)) for x in remaining]
            fair = float(ob["stake"]) * float(ob["decimal_odds"]) * float(np.prod(rem_ps)) if rem_ps else 0
            offer = fair * 0.93
            dec = eval_co(rem_ps, float(ob["decimal_odds"]), float(ob["stake"]), offer, risk_aversion=0.08)
            x1, x2, x3, x4 = st.columns(4)
            x1.metric("Offer $", f"{offer:.2f}")
            x2.metric("Ride EV $", f"{dec.ride_ev:.2f}")
            x3.metric("Action", "CASH" if dec.recommend_cashout else "RIDE")
            if x4.button("Cash out", key=f"cash_{ob['id']}"):
                hist.cashout(ob["id"], offer, float(ob["stake"]))
                st.toast("Cashed", icon="✓")
                st.rerun()
            c1, c2 = st.columns(2)
            if c1.button("Mark won (full)", key=f"win_{ob['id']}"):
                pnl = float(ob["stake"]) * (float(ob["decimal_odds"]) - 1)
                # optional close line entry
                hist.settle(ob["id"], True, pnl, close_decimal=float(ob["decimal_odds"]) * 0.98)
                st.rerun()
            if c2.button("Mark lost", key=f"lose_{ob['id']}"):
                hist.mark_leg_lost(ob["id"], float(ob["stake"]))
                st.rerun()
        st.divider()

with tab_path:
    path = data.get("path")
    if path:
        top = path["ticket"]
        st.markdown(f"**Simulated ticket** ({top.n_legs} leg · {top.source})")
        st.code(top.id[:100])
        a, b, c, d = st.columns(4)
        a.metric("Mean PnL @ $50", f"${path['mean_pnl']:.2f}")
        b.metric("P(profit)", f"{100*path['p_profit']:.1f}%")
        c.metric("P(mid cash)", f"{100*path['p_cashout']:.1f}%")
        d.metric("P(full win)", f"{100*path['p_full_win']:.1f}%")
        st.caption("Policy: cash when offer ≥ model ride EV (risk-adjusted).")
    else:
        st.caption("Need a 2+ leg combo on the board for path simulation.")

with tab_hist:
    st.markdown("### Closing line value")
    st.caption("CLV > 0 ⇒ you got a better price than the close. Persistently positive CLV is the skill signal.")
    s = hist.stats()
    h1, h2, h3 = st.columns(3)
    h1.metric("Avg CLV", f"{s['avg_clv']:+.4f}" if s["avg_clv"] is not None else "—")
    h2.metric("% positive CLV", f"{100*s['positive_clv_rate']:.0f}%" if s["positive_clv_rate"] is not None else "—")
    h3.metric("Settled n", s["n"])
    if s.get("avg_slippage") is not None:
        st.caption(f"Avg execution slippage: {s['avg_slippage']:+.4f} (decimal odds)")
    st.caption(f"Kelly gate: {'OPEN' if gate.allowed_to_scale else 'CLOSED'} — {gate.reason}")
    recent = hist.list_recent(40)
    if recent:
        st.dataframe(pd.DataFrame([{
            "id": r["id"],
            "when": r["created_at"],
            "status": r["status"],
            "stake": r["stake"],
            "odds": round(r["decimal_odds"] or 0, 2),
            "p": round(r["model_p"] or 0, 3),
            "pnl": r["pnl"],
            "clv": r["clv"],
            "desc": (r["description"] or "")[:50],
        } for r in recent]), use_container_width=True, hide_index=True)
    else:
        st.caption("No settled bets yet.")

with tab_legs:
    if data["ranked"]:
        st.dataframe(pd.DataFrame([{
            "src": "PM" if l.sport == "PRED" else l.sport,
            "selection": l.description[:70],
            "p": round(l.p, 3),
            "odds": round(l.decimal_odds, 2),
            "edge": round(l.edge, 3),
            "EV": round(l.ev, 4),
        } for l in data["ranked"]]), use_container_width=True, hide_index=True)
