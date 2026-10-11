"""NFL team stats bot — efficiency proxies (synthetic until paid feed).

Replace payload with real EPA/DVOA/success rate when you wire a stats API.
"""

from __future__ import annotations
from typing import List
import numpy as np
from sports_network.bots.base import BaseBot, BotRecord

TEAMS = ["KC","BUF","BAL","DET","PHI","SF","DAL","MIA","CIN","GB",
         "HOU","LAR","MIN","SEA","TB","ATL","LAC","PIT","CLE","JAX",
         "IND","NO","CHI","NYJ","LV","TEN","DEN","ARI","WAS","NYG","NE","CAR"]

class NFLStatsBot(BaseBot):
    name = "nfl_stats_bot"
    domain = "stats"
    sport = "NFL"

    def collect(self) -> List[BotRecord]:
        rng = np.random.default_rng(42)
        out = []
        for t in TEAMS:
            off = float(rng.normal(0.0, 0.12))
            de = float(rng.normal(0.0, 0.12))
            out.append(BotRecord(
                domain="stats", sport="NFL", entity_id=t,
                source="synth-epa-proxy", confidence=0.5,
                payload={
                    "team": t,
                    "off_epa_proxy": round(off, 3),
                    "def_epa_proxy": round(de, 3),
                    "net_epa_proxy": round(off - de, 3),
                    "pace": round(float(rng.uniform(58, 68)), 1),
                },
            ))
        return out
