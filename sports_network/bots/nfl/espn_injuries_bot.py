"""NFL injuries from ESPN public team endpoints — no API key."""

from __future__ import annotations
from typing import List
try:
    import requests
except ImportError:
    requests = None

from sports_network.bots.base import BaseBot, BotRecord

# ESPN team ids (subset of 32)
NFL_TEAM_IDS = {
    "ARI": 22, "ATL": 1, "BAL": 33, "BUF": 2, "CAR": 29, "CHI": 3, "CIN": 4, "CLE": 5,
    "DAL": 6, "DEN": 7, "DET": 8, "GB": 9, "HOU": 34, "IND": 11, "JAX": 30, "KC": 12,
    "LAC": 24, "LAR": 14, "LV": 13, "MIA": 15, "MIN": 16, "NE": 17, "NO": 18, "NYG": 19,
    "NYJ": 20, "PHI": 21, "PIT": 23, "SEA": 26, "SF": 25, "TB": 27, "TEN": 10, "WAS": 28,
}

UA = {"User-Agent": "EVDeskSportsNetwork/1.0"}
STATUS_ELO = {
    "out": -45, "doubtful": -25, "questionable": -10,
    "injury reserve": -40, "ir": -40, "pup": -30,
}


class ESPNInjuriesBot(BaseBot):
    name = "espn_injuries_bot"
    domain = "injuries"
    sport = "NFL"

    def collect(self) -> List[BotRecord]:
        if not requests:
            return []
        out = []
        team_delta = {}
        # Sample teams to stay fast; expand to all 32 when scheduled
        for abbr, tid in list(NFL_TEAM_IDS.items())[:16]:
            try:
                r = requests.get(
                    f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/{tid}/injuries",
                    headers=UA, timeout=6,
                )
                if r.status_code != 200:
                    continue
                data = r.json() or {}
                items = data.get("injuries") or data.get("items") or []
                # structure varies; also try athletes
                if not items and "athletes" in data:
                    items = data["athletes"]
                delta = 0.0
                notes = []
                for it in items[:8]:
                    st = str(it.get("status") or it.get("type") or "").lower()
                    name = it.get("displayName") or it.get("name") or "?"
                    for k, v in STATUS_ELO.items():
                        if k in st:
                            delta += v * 0.35  # dampen multi-player stack
                            notes.append(f"{name}:{st}")
                            break
                team_delta[abbr] = delta
                out.append(BotRecord(
                    domain="injuries", sport="NFL", entity_id=abbr,
                    source="espn-public", confidence=0.6 if notes else 0.3,
                    payload={"team": abbr, "elo_delta": round(delta, 1), "notes": notes[:5]},
                ))
            except Exception:
                continue

        # push into model
        try:
            from core.nfl_model import get_nfl_model
            m = get_nfl_model()
            for t, d in team_delta.items():
                if d:
                    m.set_injury(t, d)
        except Exception:
            pass
        return out
