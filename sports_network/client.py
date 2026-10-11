"""
Client for our Sports Network API (or direct store access).
EV Desk uses this instead of depending only on third-party keys.
"""

from __future__ import annotations
import os
from typing import Any, Dict, List, Optional

try:
    import requests
except ImportError:
    requests = None

from sports_network.storage.store import NetworkStore


class NetworkClient:
    def __init__(self, base_url: Optional[str] = None, use_store: bool = True):
        self.base = (base_url or os.environ.get("SPORTS_NETWORK_URL") or "http://127.0.0.1:8787").rstrip("/")
        self.use_store = use_store
        self.store = NetworkStore() if use_store else None

    def health(self) -> dict:
        if requests:
            try:
                r = requests.get(f"{self.base}/health", timeout=3)
                if r.ok:
                    return r.json()
            except Exception:
                pass
        return {"status": "store-direct" if self.store else "down"}

    def ratings(self, sport: str = "NFL") -> List[dict]:
        if self.store:
            return self.store.query("ratings", sport)
        if requests:
            r = requests.get(f"{self.base}/ratings/{sport}", timeout=10)
            return (r.json() or {}).get("ratings") or []
        return []

    def weather(self, sport: str = "NFL") -> List[dict]:
        if self.store:
            return self.store.query("weather", sport)
        if requests:
            r = requests.get(f"{self.base}/weather/{sport}", timeout=10)
            return (r.json() or {}).get("weather") or []
        return []

    def injuries(self) -> List[dict]:
        if self.store:
            return self.store.query("injuries", "NFL")
        if requests:
            r = requests.get(f"{self.base}/injuries", timeout=10)
            return (r.json() or {}).get("injuries") or []
        return []

    def stats(self, sport: str = "NFL") -> List[dict]:
        if self.store:
            return self.store.query("stats", sport)
        return []

    def odds(self, sport: Optional[str] = None) -> List[dict]:
        if self.store:
            return self.store.query("odds", sport, limit=50)
        return []

    def summary(self) -> dict:
        if self.store:
            return self.store.summary()
        if requests:
            try:
                return requests.get(f"{self.base}/summary", timeout=5).json()
            except Exception:
                pass
        return {}

    def collect(self) -> dict:
        """Run bot team locally (preferred) or POST /collect."""
        from sports_network.orchestrator import run_all
        return run_all(self.store)
