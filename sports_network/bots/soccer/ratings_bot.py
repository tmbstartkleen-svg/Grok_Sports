"""
Soccer ratings bot — internal Elo with optional result updates.

No paid API. Writes domain=ratings sport=SOCCER.
If data/soccer_results.json exists (list of {home, away, home_goals, away_goals}),
Elo is updated from those results before publishing.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict
from sports_network.bots.base import BaseBot, BotRecord

ROOT = Path(__file__).resolve().parents[3]
RESULTS_PATH = ROOT / "data" / "soccer_results.json"

SEED: Dict[str, float] = {
    "MCI": 1850, "ARS": 1820, "LIV": 1810, "RMA": 1840, "BAR": 1800,
    "BAY": 1830, "PSG": 1790, "INT": 1780, "ATM": 1760, "CHE": 1750,
    "TOT": 1740, "NAP": 1730, "JUV": 1720, "BVB": 1710, "ACM": 1700,
    "MUN": 1690, "NEW": 1680, "SEV": 1670, "RBL": 1660, "AVL": 1650,
}

K = 20.0
SCALE = 400.0
HFA = 40.0


def _expected(rh: float, ra: float) -> float:
    return 1.0 / (1.0 + 10 ** (-((rh + HFA) - ra) / SCALE))


def _update(ratings: Dict[str, float], home: str, away: str, hg: int, ag: int) -> None:
    home, away = home.upper(), away.upper()
    if home not in ratings:
        ratings[home] = 1500.0
    if away not in ratings:
        ratings[away] = 1500.0
    rh, ra = ratings[home], ratings[away]
    exp_h = _expected(rh, ra)
    if hg > ag:
        score_h = 1.0
    elif hg < ag:
        score_h = 0.0
    else:
        score_h = 0.5
    gd = abs(hg - ag)
    mult = min(1.5, 1.0 + 0.1 * max(0, gd - 1))
    ratings[home] = rh + K * mult * (score_h - exp_h)
    ratings[away] = ra + K * mult * ((1.0 - score_h) - (1.0 - exp_h))


class SoccerRatingsBot(BaseBot):
    name = "soccer_ratings_bot"
    domain = "ratings"
    sport = "SOCCER"

    def collect(self) -> List[BotRecord]:
        ratings = dict(SEED)
        n_results = 0
        if RESULTS_PATH.exists():
            try:
                data = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
                for row in data if isinstance(data, list) else []:
                    home = str(row.get("home") or "").upper()
                    away = str(row.get("away") or "").upper()
                    hg = int(row.get("home_goals", row.get("hg", 0)))
                    ag = int(row.get("away_goals", row.get("ag", 0)))
                    if home and away:
                        _update(ratings, home, away, hg, ag)
                        n_results += 1
            except Exception as e:
                print(f"[soccer_ratings_bot] results skip: {e}")

        out = []
        for club, elo in sorted(ratings.items(), key=lambda x: -x[1]):
            out.append(BotRecord(
                domain="ratings",
                sport="SOCCER",
                entity_id=club,
                source="internal-elo" + ("+results" if n_results else "-seed"),
                confidence=0.70 if n_results else 0.55,
                payload={
                    "club": club,
                    "elo": round(elo, 1),
                    "results_applied": n_results,
                    "note": "update via data/soccer_results.json [{home,away,home_goals,away_goals}]",
                },
            ))
        return out
