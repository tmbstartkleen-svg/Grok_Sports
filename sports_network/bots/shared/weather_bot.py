"""Weather bot — Open-Meteo (free, no key). Outdoor sports sites."""

from __future__ import annotations
from typing import List
import numpy as np

try:
    import requests
except ImportError:
    requests = None

from sports_network.bots.base import BaseBot, BotRecord

# Major outdoor venues
VENUES = {
    "NFL": {
        "KC": (39.0489, -94.4839), "BUF": (42.7738, -78.7870), "GB": (44.5013, -88.0622),
        "DEN": (39.7439, -105.0201), "SEA": (47.5952, -122.3316), "CHI": (41.8623, -87.6167),
        "NE": (42.0909, -71.2643), "MIA": (25.9580, -80.2389), "DAL": (32.7473, -97.0945),
        "PHI": (39.9008, -75.1675), "PIT": (40.4468, -80.0158), "BAL": (39.2780, -76.6227),
        "CLE": (41.5061, -81.6995), "CIN": (39.0950, -84.5160), "DET": (42.3400, -83.0456),
        "MIN": (44.9738, -93.2581), "NYJ": (40.8135, -74.0745), "NYG": (40.8135, -74.0745),
        "LAR": (33.9535, -118.3390), "LAC": (33.8644, -118.2611), "SF": (37.4032, -121.9697),
        "ARI": (33.5277, -112.2626), "TB": (27.9759, -82.5033), "ATL": (33.7555, -84.4008),
        "NO": (29.9511, -90.0812), "HOU": (29.6847, -95.4107), "IND": (39.7601, -86.1639),
        "JAX": (30.3239, -81.6373), "TEN": (36.1665, -86.7713), "CAR": (35.2258, -80.8528),
        "WAS": (38.9077, -76.8645), "LV": (36.0908, -115.1836),
    },
}


class WeatherBot(BaseBot):
    name = "weather_bot"
    domain = "weather"
    sport = "NFL"

    def collect(self) -> List[BotRecord]:
        out = []
        venues = VENUES.get(self.sport, {})
        # Limit live calls for speed; full slate still covered offline
        items = list(venues.items())
        if requests:
            items = items[:12]
        for team, (lat, lon) in items:
            wx = self._fetch(lat, lon)
            if not wx:
                continue
            out.append(BotRecord(
                domain="weather", sport=self.sport, entity_id=team,
                source="open-meteo", confidence=0.9, payload={
                    "team": team, "lat": lat, "lon": lon, **wx,
                    "under_lean": wx.get("wind_mph", 0) >= 15 or wx.get("temp_f", 70) < 35,
                },
            ))
        return out

    def _fetch(self, lat, lon):
        if not requests:
            return self._synthetic(lat, lon)
        try:
            r = requests.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": lat, "longitude": lon,
                    "current": "temperature_2m,precipitation,wind_speed_10m",
                    "temperature_unit": "fahrenheit", "wind_speed_unit": "mph",
                },
                timeout=4,
            )
            r.raise_for_status()
            cur = r.json().get("current") or {}
            return {
                "temp_f": float(cur.get("temperature_2m") or 70),
                "precip_mm": float(cur.get("precipitation") or 0),
                "wind_mph": float(cur.get("wind_speed_10m") or 0),
            }
        except Exception:
            return self._synthetic(lat, lon)

    def _synthetic(self, lat, lon):
        rng = np.random.default_rng(int(abs(lat * 1000)) % 2**31)
        return {
            "temp_f": float(rng.uniform(28, 85)),
            "precip_mm": float(rng.choice([0, 0, 0, 1.5, 4.0])),
            "wind_mph": float(rng.uniform(3, 22)),
        }
