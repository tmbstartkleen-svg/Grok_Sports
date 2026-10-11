"""NFL ratings bot — publishes Elo snapshot from our model."""

from __future__ import annotations
from typing import List
from sports_network.bots.base import BaseBot, BotRecord

class NFLRatingsBot(BaseBot):
    name = "nfl_ratings_bot"
    domain = "ratings"
    sport = "NFL"

    def collect(self) -> List[BotRecord]:
        try:
            from core.nfl_model import get_nfl_model, DEFAULT_RATINGS
        except ImportError:
            from sports_ev_engine.core.nfl_model import get_nfl_model, DEFAULT_RATINGS  # type: ignore
        model = get_nfl_model()
        out = []
        for team, rating in sorted(model.ratings.items(), key=lambda x: -x[1]):
            out.append(BotRecord(
                domain="ratings", sport="NFL", entity_id=team,
                source="evdesk-elo", confidence=0.85,
                payload={
                    "team": team,
                    "elo": rating,
                    "injury_elo": model.injury_elo.get(team, 0.0),
                    "rest_days": model.rest_days.get(team, 7.0),
                    "effective_elo": model.elo(team),
                },
            ))
        return out
