"""
Elite factor adjustments on top of consensus market prices.

Home-field advantage, weather (totals / outdoor sports), and injury
modifiers shift model_p before Kelly / parlay construction.

Weather uses Open-Meteo (free, no key) when coordinates are known.
Injuries are applied as manual or feed-driven deltas (API optional).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Any
import math

try:
    import requests
except ImportError:
    requests = None


# Approximate HFA in probability points (moneyline scale) by sport
HFA_PROB = {
    "NFL": 0.03,
    "NCAAF": 0.035,
    "NBA": 0.03,
    "NHL": 0.02,
    "MLB": 0.02,
    "EPL": 0.04,
    "MLS": 0.035,
}

# Rough stadium coords for weather (extend as needed)
STADIUM_COORDS: Dict[str, tuple] = {
    # city/team keywords → (lat, lon)
    "kansas city": (39.0489, -94.4839),
    "buffalo": (42.7738, -78.7870),
    "green bay": (44.5013, -88.0622),
    "denver": (39.7439, -105.0201),
    "seattle": (47.5952, -122.3316),
    "chicago": (41.8623, -87.6167),
    "boston": (42.3467, -71.0972),
    "miami": (25.9580, -80.2389),
    "dallas": (32.7473, -97.0945),
    "philadelphia": (39.9008, -75.1675),
    "pittsburgh": (40.4468, -80.0158),
    "baltimore": (39.2780, -76.6227),
    "cleveland": (41.5061, -81.6995),
    "cincinnati": (39.0950, -84.5160),
    "detroit": (42.3400, -83.0456),
    "minneapolis": (44.9738, -93.2581),
    "new york": (40.8135, -74.0745),
    "los angeles": (33.9535, -118.3390),
    "san francisco": (37.4032, -121.9697),
    "arizona": (33.5277, -112.2626),
    "tampa": (27.9759, -82.5033),
    "atlanta": (33.7555, -84.4008),
    "new orleans": (29.9511, -90.0812),
    "houston": (29.6847, -95.4107),
    "indianapolis": (39.7601, -86.1639),
    "jacksonville": (30.3239, -81.6373),
    "tennessee": (36.1665, -86.7713),
    "nashville": (36.1665, -86.7713),
    "carolina": (35.2258, -80.8528),
    "washington": (38.9077, -76.8645),
    "las vegas": (36.0908, -115.1836),
}


@dataclass
class FactorAdjustment:
    delta_p: float = 0.0
    notes: List[str] = field(default_factory=list)


def home_away_adjustment(
    selection: str,
    home: str,
    away: str,
    sport: str,
    is_home_bet: Optional[bool] = None,
) -> FactorAdjustment:
    """
    Shift model_p for home/away moneyline sides.
    Markets already price HFA; we only apply a *residual* edge when our
    HFA prior differs from a flat 50/50 after de-vig — kept small.
    """
    hfa = HFA_PROB.get(sport, 0.025)
    adj = FactorAdjustment()
    sel_l = selection.lower()
    home_l = (home or "").lower()
    away_l = (away or "").lower()

    if is_home_bet is None:
        if home_l and home_l in sel_l:
            is_home_bet = True
        elif away_l and away_l in sel_l:
            is_home_bet = False
        else:
            return adj

    # Residual: markets embed most HFA; only nudge ±0.5× published HFA as prior strength
    residual = 0.35 * hfa
    if is_home_bet:
        adj.delta_p = residual * 0.15  # tiny home lean after market pricing
        adj.notes.append(f"HFA residual +{adj.delta_p:.3f} ({sport})")
    else:
        adj.delta_p = -residual * 0.15
        adj.notes.append(f"Road residual {adj.delta_p:.3f} ({sport})")
    return adj


def weather_adjustment(
    sport: str,
    home: str,
    wind_mph: Optional[float] = None,
    temp_f: Optional[float] = None,
    precip_mm: Optional[float] = None,
    is_total_market: bool = False,
) -> FactorAdjustment:
    """
    Outdoor sports: wind/cold/rain affect totals more than sides.
    Returns delta on model_p for the selection (over gets negative in high wind).
    """
    adj = FactorAdjustment()
    outdoor = sport in ("NFL", "NCAAF", "MLB", "MLS", "EPL")
    if not outdoor:
        return adj

    # Auto-fetch if not provided
    if wind_mph is None or temp_f is None:
        coords = _lookup_coords(home)
        if coords and requests:
            w = fetch_weather(coords[0], coords[1])
            if w:
                wind_mph = w.get("wind_mph", wind_mph)
                temp_f = w.get("temp_f", temp_f)
                precip_mm = w.get("precip_mm", precip_mm)

    if wind_mph is None and temp_f is None:
        return adj

    if is_total_market:
        # High wind / cold → under lean
        if wind_mph and wind_mph >= 15:
            d = -0.02 - 0.01 * max(0, (wind_mph - 15) / 10)
            adj.delta_p += d
            adj.notes.append(f"Wind {wind_mph:.0f}mph → under lean {d:.3f}")
        if temp_f is not None and temp_f < 35:
            d = -0.015
            adj.delta_p += d
            adj.notes.append(f"Cold {temp_f:.0f}°F → under lean {d:.3f}")
        if precip_mm and precip_mm > 2:
            d = -0.01
            adj.delta_p += d
            adj.notes.append(f"Precip {precip_mm:.1f}mm → under lean")
    else:
        # Sides: extreme weather slight dog lean (variance), very small
        if wind_mph and wind_mph >= 20:
            adj.delta_p += -0.005
            adj.notes.append(f"High wind variance drag {adj.delta_p:.3f}")

    return adj


def injury_adjustment(
    selection: str,
    injury_delta_by_team: Optional[Dict[str, float]] = None,
) -> FactorAdjustment:
    """
    injury_delta_by_team: team name/abbr → probability points to ADD to that team's ML.
    Negative = key player out hurts the team.
    Example: {"BUF": -0.04, "KC": 0.0} when Allen is out.
    """
    adj = FactorAdjustment()
    if not injury_delta_by_team:
        return adj
    sel_l = selection.lower()
    for team, delta in injury_delta_by_team.items():
        if team.lower() in sel_l:
            adj.delta_p += float(delta)
            adj.notes.append(f"Injury factor {team}: {delta:+.3f}")
    return adj


def combine_adjustments(*adjs: FactorAdjustment, cap: float = 0.06) -> FactorAdjustment:
    total = FactorAdjustment()
    for a in adjs:
        total.delta_p += a.delta_p
        total.notes.extend(a.notes)
    total.delta_p = max(-cap, min(cap, total.delta_p))
    return total


def apply_factors_to_p(base_p: float, adj: FactorAdjustment) -> float:
    p = base_p + adj.delta_p
    return float(max(0.05, min(0.95, p)))


def _lookup_coords(home: str) -> Optional[tuple]:
    if not home:
        return None
    h = home.lower()
    for key, coords in STADIUM_COORDS.items():
        if key in h:
            return coords
    return None


def fetch_weather(lat: float, lon: float) -> Optional[Dict[str, float]]:
    """Open-Meteo current weather – free, no API key."""
    if not requests:
        return None
    try:
        url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            "&current=temperature_2m,precipitation,wind_speed_10m"
            "&temperature_unit=fahrenheit&wind_speed_unit=mph"
        )
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        cur = r.json().get("current", {})
        return {
            "temp_f": float(cur.get("temperature_2m") or 70),
            "precip_mm": float(cur.get("precipitation") or 0),
            "wind_mph": float(cur.get("wind_speed_10m") or 0),
        }
    except Exception:
        return None
