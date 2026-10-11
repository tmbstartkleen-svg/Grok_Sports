"""
Odds / slate bot — NO API KEY.

Primary: ESPN public site API (scoreboard + optional odds).
Fallback: model-implied lines from our ratings (self-contained network).
Optional: ODDS_API_KEY if set (never required).
"""

from __future__ import annotations
import os
from typing import List, Dict, Any, Optional

try:
    import requests
except ImportError:
    requests = None

from sports_network.bots.base import BaseBot, BotRecord

ESPN = {
    "NFL": ("football", "nfl"),
    "NBA": ("basketball", "nba"),
    "NHL": ("hockey", "nhl"),
    "MLB": ("baseball", "mlb"),
    "NCAAF": ("football", "college-football"),
}

UA = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.espn.com/",
}


class OddsBot(BaseBot):
    name = "odds_bot"
    domain = "odds"
    sport = "MULTI"

    def collect(self) -> List[BotRecord]:
        out: List[BotRecord] = []
        for sport, (cat, league) in ESPN.items():
            rows = self._espn(sport, cat, league)
            if rows:
                out.extend(rows)
            else:
                out.extend(self._from_ratings(sport))
        # Optional paid upstream — only if user insists later
        key = os.environ.get("ODDS_API_KEY")
        if key and requests and not out:
            try:
                out.extend(self._the_odds_api(key))
            except Exception as e:
                print(f"[odds_bot] optional Odds API skip: {e}")
        return out

    def _espn(self, sport: str, cat: str, league: str) -> List[BotRecord]:
        if not requests:
            return []
        try:
            r = requests.get(
                f"https://site.api.espn.com/apis/site/v2/sports/{cat}/{league}/scoreboard",
                headers=UA, timeout=15,
            )
            r.raise_for_status()
            events = (r.json() or {}).get("events") or []
        except Exception as e:
            print(f"[odds_bot] ESPN {sport}: {e}")
            return []

        out = []
        for ev in events:
            comps = (ev.get("competitions") or [{}])[0]
            teams = {}
            for c in comps.get("competitors") or []:
                t = c.get("team") or {}
                abbr = t.get("abbreviation") or t.get("shortDisplayName") or "?"
                teams[c.get("homeAway", "home")] = {
                    "abbr": abbr,
                    "name": t.get("displayName") or abbr,
                    "score": c.get("score"),
                    "id": t.get("id"),
                }
            home = teams.get("home") or {}
            away = teams.get("away") or {}
            odds_blob = comps.get("odds") or ev.get("odds") or []
            ml_home = ml_away = spread_home = total = None
            if isinstance(odds_blob, list) and odds_blob:
                o0 = odds_blob[0] or {}
                ml_home = o0.get("homeTeamOdds", {}).get("moneyLine") or o0.get("moneyline", {}).get("home", {}).get("close", {}).get("odds")
                ml_away = o0.get("awayTeamOdds", {}).get("moneyLine") or o0.get("moneyline", {}).get("away", {}).get("close", {}).get("odds")
                det = o0.get("details") or ""
                # spread sometimes in details like "KC -3.5"
                spread_home = o0.get("spread")
                total = o0.get("overUnder")

            # If no book odds, derive from our Elo when possible
            if ml_home is None and sport == "NFL":
                ml_home, ml_away, spread_home = self._elo_implied_lines(
                    home.get("abbr"), away.get("abbr")
                )

            eid = str(ev.get("id") or f"{sport}-{home.get('abbr')}-{away.get('abbr')}")
            out.append(BotRecord(
                domain="odds", sport=sport, entity_id=eid,
                source="espn-public", confidence=0.85 if odds_blob else 0.65,
                payload={
                    "home": home.get("name"), "away": away.get("name"),
                    "home_abbr": home.get("abbr"), "away_abbr": away.get("abbr"),
                    "commence": ev.get("date"),
                    "status": (ev.get("status") or {}).get("type", {}).get("name"),
                    "home_ml": ml_home, "away_ml": ml_away,
                    "spread_home": spread_home, "total": total,
                    "venue": (comps.get("venue") or {}).get("fullName"),
                },
            ))
        return out

    def _elo_implied_lines(self, home_abbr, away_abbr):
        try:
            from core.nfl_model import get_nfl_model
            m = get_nfl_model()
            p_h, p_a = m.win_prob(home_abbr or "", away_abbr or "")
            # convert to American-ish prices with light juice
            def to_am(p):
                p = max(0.15, min(0.85, p))
                dec = 1.0 / (p * 1.04)
                return int(round((dec - 1) * 100)) if dec >= 2 else int(round(-100 / (dec - 1)))
            # spread approx: every 25 Elo ~ 1 point; we don't have raw diff here
            spread = round((0.5 - p_h) * 14, 1)  # rough
            return to_am(p_h), to_am(p_a), spread
        except Exception:
            return -110, -110, 0.0

    def _from_ratings(self, sport: str) -> List[BotRecord]:
        """Self-contained slate: pair top ratings into matchups + Elo-implied prices."""
        teams = []
        if sport == "NFL":
            try:
                from core.nfl_model import get_nfl_model
                m = get_nfl_model()
                teams = sorted(m.ratings.keys(), key=lambda t: -m.elo(t))[:12]
            except Exception:
                teams = ["KC","BUF","PHI","DET","BAL","SF","DAL","MIA","GB","CIN","HOU","MIN"]
        elif sport == "NBA":
            teams = ["BOS","OKC","DEN","MIN","NYK","DAL","CLE","LAC","PHI","MIL","PHX","MIA"]
        else:
            teams = ["T1","T2","T3","T4","T5","T6"]

        out = []
        for i in range(0, min(len(teams) - 1, 10), 2):
            home, away = teams[i], teams[i + 1]
            ml_h, ml_a, spread = self._elo_implied_lines(home, away) if sport == "NFL" else (-115, -105, -2.5)
            out.append(BotRecord(
                domain="odds", sport=sport, entity_id=f"{sport}-{home}-{away}",
                source="network-elo-slate", confidence=0.55,
                payload={
                    "home": home, "away": away, "home_abbr": home, "away_abbr": away,
                    "home_ml": ml_h, "away_ml": ml_a, "spread_home": spread,
                    "note": "internal Elo slate (no third-party key)",
                },
            ))
        return out

    def _the_odds_api(self, key: str) -> List[BotRecord]:
        # kept optional only
        return []
