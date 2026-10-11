"""
Push Sports Network snapshots into EV core models before scan.
"""

from __future__ import annotations
from typing import Dict
from sports_network.client import NetworkClient


def apply_network_to_models(client: NetworkClient | None = None) -> Dict[str, int]:
    client = client or NetworkClient()
    stats = {"injuries": 0, "weather": 0, "ratings": 0}
    try:
        from core.nfl_model import get_nfl_model
        model = get_nfl_model()
    except Exception:
        return stats

    for row in client.injuries():
        p = row.get("payload") or {}
        team = p.get("team") or row.get("entity_id")
        delta = p.get("elo_delta", 0)
        if team and delta:
            model.set_injury(str(team), float(delta))
            stats["injuries"] += 1

    # weather stored for factors / UI; optional rest not derived here
    stats["weather"] = len(client.weather("NFL"))
    stats["ratings"] = len(client.ratings("NFL"))
    return stats


def network_feature_bundle(client: NetworkClient | None = None) -> dict:
    client = client or NetworkClient()
    return {
        "summary": client.summary(),
        "top_nfl": sorted(
            [
                (r.get("entity_id"), (r.get("payload") or {}).get("effective_elo")
                 or (r.get("payload") or {}).get("elo"))
                for r in client.ratings("NFL")
            ],
            key=lambda x: -(x[1] or 0),
        )[:10],
        "weather_alerts": [
            r for r in client.weather("NFL")
            if (r.get("payload") or {}).get("under_lean")
        ][:8],
        "injuries": client.injuries(),
    }
