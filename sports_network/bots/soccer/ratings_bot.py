"""
Soccer ratings bot — lightweight internal Elo seed for major clubs.
No paid API. Writes domain=ratings sport=SOCCER.
"""

from __future__ import annotations
from typing import List
from sports_network.bots.base import BaseBot, BotRecord

# Seed Elo for a handful of top clubs (illustrative; extend freely)
SEED = {
    "MCI": 1850, "ARS": 1820, "LIV": 1810, "RMA": 1840, "BAR": 1800,
    "BAY": 1830, "PSG": 1790, "INT": 1780, "ATM": 1760, "CHE": 1750,
    "TOT": 1740, "NAP": 1730, "JUV": 1720, "BVB": 1710, "ACM": 1700,
}


class SoccerRatingsBot(BaseBot):
    name = "soccer_ratings_bot"
    domain = "ratings"
    sport = "SOCCER"

    def collect(self) -> List[BotRecord]:
        out = []
        for club, elo in SEED.items():
            out.append(BotRecord(
                domain="ratings",
                sport="SOCCER",
                entity_id=club,
                source="internal-elo-seed",
                confidence=0.55,
                payload={"club": club, "elo": elo, "note": "seed ratings — refine with results"},
            ))
        return out
