"""
Base collector bot. Each bot owns one data domain and writes normalized records
into the sports network store. Bots are the supply; the local API is the key.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import json


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class BotRecord:
    domain: str          # odds | weather | injuries | stats | ratings | news
    sport: str
    entity_id: str       # game_id, team, player
    payload: Dict[str, Any]
    source: str
    collected_at: str = field(default_factory=utc_now)
    confidence: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)


class BaseBot(ABC):
    name: str = "base"
    domain: str = "generic"
    sport: str = "MULTI"

    def __init__(self, store: "NetworkStore"):
        self.store = store

    @abstractmethod
    def collect(self) -> List[BotRecord]:
        ...

    def run(self) -> int:
        records = self.collect()
        n = 0
        for r in records:
            self.store.write(r)
            n += 1
        self.store.log_run(self.name, self.domain, self.sport, n)
        return n
