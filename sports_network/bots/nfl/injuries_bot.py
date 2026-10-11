"""NFL injuries bot — structured injury feed (manual/API/synthetic).

Plug a real feed later (e.g. ESPN scrape or paid API). Until then,
accepts env NFL_INJURIES=BUF:-40,KC:-15 (Elo deltas) and publishes them.
"""

from __future__ import annotations
import os
from typing import List
from sports_network.bots.base import BaseBot, BotRecord

class NFLInjuriesBot(BaseBot):
    name = "nfl_injuries_bot"
    domain = "injuries"
    sport = "NFL"

    def collect(self) -> List[BotRecord]:
        out = []
        raw = os.environ.get("NFL_INJURIES", "")
        # also push into live model
        try:
            from core.nfl_model import get_nfl_model
            model = get_nfl_model()
        except Exception:
            model = None

        pairs = []
        for part in raw.split(","):
            part = part.strip()
            if ":" not in part:
                continue
            team, delta = part.split(":", 1)
            try:
                pairs.append((team.strip().upper(), float(delta.strip())))
            except ValueError:
                continue

        if not pairs:
            # neutral snapshot so network still has the domain
            pairs = [("BUF", 0.0), ("KC", 0.0)]

        for team, delta in pairs:
            if model is not None and delta != 0:
                model.set_injury(team, delta)
            out.append(BotRecord(
                domain="injuries", sport="NFL", entity_id=team,
                source="env-feed", confidence=0.7 if delta != 0 else 0.4,
                payload={"team": team, "elo_delta": delta, "note": "QB/skill injury proxy"},
            ))
        return out
