"""
Grok Sports Network API + Dashboard host.

  python3 -m sports_network.api.server
  # open http://127.0.0.1:8787/
"""

from __future__ import annotations
import json
import os
import sys
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sports_network.storage.store import NetworkStore
from sports_network.orchestrator import run_all

DASH_PATH = Path(__file__).resolve().parent.parent / "dashboard.html"
_BOARD_CACHE = {}


def _build_board() -> dict:
    import numpy as np
    from core import (
        score_diff_monte_carlo, expected_value, edge, Leg, rank_legs,
        markets_to_model_rows, legs_from_pred_markets, build_combo_tickets,
        BetHistory, build_action_queue, clv_gate,
    )
    from core.risk_limits import filter_actions_by_risk
    from data.prediction_markets import PredictionMarketConnector
    from sports_network.bridge import fetch_slate_with_network

    BANKROLL, KELLY, MAX_PER = 10000.0, 0.25, 0.04
    MIN_EDGE, MIN_P, SEED = 0.015, 0.45, 42
    hist = BetHistory()
    gate = clv_gate(hist, base_kelly=KELLY)
    legs, notes = [], []
    markets, net_notes, injury_deltas = fetch_slate_with_network(
        sports=["NFL", "NCAAF", "NBA", "NHL"], days_ahead=5, seed=SEED, prefer_network=True
    )
    notes.extend(net_notes)
    rows = markets_to_model_rows(
        markets, prefer_books=None, apply_weather=True, apply_hfa=True,
        injury_deltas=injury_deltas or None,
    )
    for i, r in enumerate(rows):
        p = float(r["model_p"])
        dec = float(r["best_decimal"])
        logit = np.log(max(p, 1e-6) / max(1 - p, 1e-6))
        mc = score_diff_monte_carlo(mean_margin=logit * 4.0, std_margin=11.0, n_sims=2000, seed=SEED + i)
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
        pms = PredictionMarketConnector(use_live=True).fetch_markets(limit=40, min_volume=200, seed=SEED, sources=["polymarket"])
    except Exception:
        pms = []
    notes.append(f"pm:{len(pms)}")
    if pms:
        legs.extend(legs_from_pred_markets(pms, min_edge=MIN_EDGE, min_p=max(0.28, MIN_P - 0.1)))
    ranked = rank_legs(legs, min_edge=MIN_EDGE, min_p=MIN_P)[:40]
    combos = build_combo_tickets(ranked, max_legs=3, min_combined_p=0.20, min_ev=0.015, max_tickets=40)
    if gate.edge_multiplier < 1.0:
        kept = []
        for t in combos:
            if t.ev * gate.edge_multiplier >= 0.01:
                t.ev = t.ev * gate.edge_multiplier
                kept.append(t)
        combos = kept
    actions = build_action_queue(combos, BANKROLL, gate.kelly_fraction, MAX_PER, max_items=8, min_edge=MIN_EDGE * 0.8)
    try:
        actions, _ = filter_actions_by_risk(actions, hist, BANKROLL, 0.03, 0.15)
    except Exception:
        pass
    ser = [{
        "rank": a.rank, "source": a.source, "n_legs": a.n_legs, "title": a.title,
        "stake": a.stake, "model_p": a.model_p, "decimal_odds": a.decimal_odds,
        "edge": a.edge, "ev": a.ev, "book": a.book, "why": a.why, "ticket_id": a.ticket_id,
    } for a in actions]
    return {
        "notes": " · ".join(notes), "actions": ser, "n_legs": len(ranked), "n_combos": len(combos),
        "gate": {"allowed": gate.allowed_to_scale, "kelly": round(gate.kelly_fraction, 3),
                 "edge_mult": round(gate.edge_multiplier, 3), "reason": gate.reason},
    }


class Handler(BaseHTTPRequestHandler):
    store = NetworkStore()

    def log_message(self, fmt, *args):
        print(f"[api] {args[0]}")

    def _json(self, code, obj):
        body = json.dumps(obj, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _html(self, code, text):
        body = text.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        global _BOARD_CACHE
        u = urlparse(self.path)
        path = u.path.rstrip("/") or "/"
        qs = parse_qs(u.query)
        if path in ("/", "/dashboard"):
            if DASH_PATH.exists():
                return self._html(200, DASH_PATH.read_text(encoding="utf-8"))
            return self._html(200, "<h1>Grok Sports</h1><p>dashboard.html missing</p>")
        if path == "/health":
            return self._json(200, {"status": "ok", "network": "grok-sports-network"})
        if path == "/summary":
            return self._json(200, self.store.summary())
        if path == "/board":
            return self._json(200, _BOARD_CACHE or {"actions": [], "note": "POST /scan first"})
        if path == "/records":
            domain = (qs.get("domain") or [None])[0]
            sport = (qs.get("sport") or [None])[0]
            limit = int((qs.get("limit") or ["100"])[0])
            return self._json(200, {"records": self.store.query(domain=domain, sport=sport, limit=limit)})
        if path.startswith("/ratings/"):
            sport = path.split("/")[-1].upper()
            return self._json(200, {"sport": sport, "ratings": self.store.query("ratings", sport)})
        if path.startswith("/weather/"):
            sport = path.split("/")[-1].upper()
            return self._json(200, {"sport": sport, "weather": self.store.query("weather", sport)})
        if path == "/odds":
            sport = (qs.get("sport") or [None])[0]
            return self._json(200, {"odds": self.store.query("odds", sport, limit=50)})
        if path == "/injuries":
            return self._json(200, {"injuries": self.store.query("injuries", "NFL")})
        if path == "/stats":
            sport = (qs.get("sport") or ["NFL"])[0]
            return self._json(200, {"stats": self.store.query("stats", sport)})
        return self._json(404, {"error": "not found", "path": path})

    def do_POST(self):
        global _BOARD_CACHE
        path = urlparse(self.path).path.rstrip("/") or "/"
        if path == "/collect":
            try:
                results = run_all(self.store)
                return self._json(200, {"ok": True, "results": results, "summary": self.store.summary()})
            except Exception as e:
                return self._json(500, {"ok": False, "error": str(e), "trace": traceback.format_exc()[-500:]})
        if path == "/scan":
            try:
                _BOARD_CACHE = _build_board()
                return self._json(200, {"ok": True, **_BOARD_CACHE})
            except Exception as e:
                return self._json(500, {"ok": False, "error": str(e), "trace": traceback.format_exc()[-800:]})
        return self._json(404, {"error": "not found"})


def main():
    port = int(os.environ.get("NETWORK_PORT", "8787"))
    print("=" * 60)
    print("  GROK SPORTS · dashboard + API")
    print(f"  http://127.0.0.1:{port}/")
    print("  POST /collect  POST /scan  GET /board")
    print("=" * 60)
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
