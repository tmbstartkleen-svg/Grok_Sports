"""NBA ratings bot — simple seed Elo until expanded."""

from __future__ import annotations
from typing import List, Dict
from sports_network.bots.base import BaseBot, BotRecord

NBA_ELO: Dict[str, float] = {
    "BOS": 1650, "OKC": 1640, "DEN": 1620, "MIN": 1600, "NYK": 1590,
    "DAL": 1580, "CLE": 1570, "LAC": 1560, "PHI": 1550, "MIL": 1545,
    "PHX": 1530, "MIA": 1525, "IND": 1520, "SAC": 1510, "LAL": 1505,
    "NOP": 1500, "ORL": 1495, "CHI": 1485, "ATL": 1480, "GSW": 1475,
}

class NBARatingsBot(BaseBot):
    name = "nba_ratings_bot"
    domain = "ratings"
    sport = "NBA"

    def collect(self) -> List[BotRecord]:
        return [
            BotRecord(
                domain="ratings", sport="NBA", entity_id=t,
                source="evdesk-nba-elo", confidence=0.7,
                payload={"team": t, "elo": e},
            )
            for t, e in sorted(NBA_ELO.items(), key=lambda x: -x[1])
        ]
