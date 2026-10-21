"""
MLB ratings bot — internal Elo seed for all 30 clubs.
No paid API. Writes domain=ratings sport=MLB.
"""

from __future__ import annotations
from typing import List, Dict
from sports_network.bots.base import BaseBot, BotRecord

SEED: Dict[str, float] = {
    "LAD": 1580, "NYY": 1565, "ATL": 1555, "HOU": 1550, "PHI": 1545,
    "BAL": 1540, "TEX": 1530, "SD": 1525, "SEA": 1520, "TB": 1515,
    "MIN": 1510, "TOR": 1505, "MIL": 1500, "ARI": 1495, "CHC": 1490,
    "BOS": 1485, "CLE": 1480, "NYM": 1475, "STL": 1470, "CIN": 1465,
    "SF": 1460, "DET": 1455, "KC": 1450, "MIA": 1445, "PIT": 1440,
    "WSH": 1435, "LAA": 1430, "OAK": 1420, "COL": 1415, "CWS": 1405,
}


class MLBRatingsBot(BaseBot):
    name = "mlb_ratings_bot"
    domain = "ratings"
    sport = "MLB"

    def collect(self) -> List[BotRecord]:
        out = []
        for team, elo in sorted(SEED.items(), key=lambda x: -x[1]):
            out.append(BotRecord(
                domain="ratings",
                sport="MLB",
                entity_id=team,
                source="internal-elo-seed",
                confidence=0.55,
                payload={"team": team, "elo": elo, "note": "seed ratings — extend with results later"},
            ))
        return out
