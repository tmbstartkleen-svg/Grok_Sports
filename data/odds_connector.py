"""
Odds & market data connector.

When ODDS_API_KEY is set → pulls live odds from The Odds API.
Otherwise → synthetic multi-day slate for offline development.
"""

from __future__ import annotations
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict, Any
import numpy as np

try:
    import requests
except ImportError:
    requests = None


@dataclass
class MarketOdds:
    event_id: str
    sport: str
    home: str
    away: str
    commence_time: datetime
    market: str                 # "h2h", "spreads", "totals"
    selection: str
    american_odds: float
    decimal_odds: float
    book: str = "consensus"
    line: Optional[float] = None
    implied_prob: float = 0.0
    raw: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.decimal_odds <= 1.0 and self.american_odds != 0:
            if self.american_odds > 0:
                self.decimal_odds = 1.0 + self.american_odds / 100.0
            else:
                self.decimal_odds = 1.0 + 100.0 / abs(self.american_odds)
        if self.implied_prob == 0.0 and self.decimal_odds > 1.0:
            self.implied_prob = 1.0 / self.decimal_odds


# Map our short names → The Odds API sport keys
SPORT_KEYS = {
    "NFL": "americanfootball_nfl",
    "NCAAF": "americanfootball_ncaaf",
    "NBA": "basketball_nba",
    "NHL": "icehockey_nhl",
    "MLB": "baseball_mlb",
    "EPL": "soccer_epl",
    "MLS": "soccer_usa_mls",
}


def american_to_decimal(american: float) -> float:
    if american > 0:
        return 1.0 + american / 100.0
    return 1.0 + 100.0 / abs(american)


class OddsConnector:
    """
    Set env ODDS_API_KEY to use The Odds API live data.
    """

    BASE = "https://api.the-odds-api.com/v4"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("ODDS_API_KEY")

    def fetch_slate(
        self,
        sports: Optional[List[str]] = None,
        days_ahead: int = 7,
        seed: int = 42,
        regions: str = "us",
        markets: str = "h2h",
        odds_format: str = "american",
    ) -> List[MarketOdds]:
        if self.api_key:
            try:
                return self._fetch_live(
                    sports or ["NFL", "NBA", "NHL", "NCAAF"],
                    days_ahead=days_ahead,
                    regions=regions,
                    markets=markets,
                    odds_format=odds_format,
                )
            except Exception as e:
                print(f"[OddsConnector] Live fetch failed ({e}); falling back to synthetic.")
                return self._synthetic_slate(
                    sports or ["NFL", "NBA", "NHL", "MLB", "NCAAF"], days_ahead, seed
                )
        return self._synthetic_slate(
            sports or ["NFL", "NBA", "NHL", "MLB", "NCAAF"], days_ahead, seed
        )

    def _fetch_live(
        self,
        sports: List[str],
        days_ahead: int = 7,
        regions: str = "us",
        markets: str = "h2h",
        odds_format: str = "american",
    ) -> List[MarketOdds]:
        if requests is None:
            raise RuntimeError("requests package required for live odds")

        now = datetime.now(timezone.utc)
        cutoff = now + timedelta(days=days_ahead)
        out: List[MarketOdds] = []

        for sport in sports:
            key = SPORT_KEYS.get(sport, sport)
            url = f"{self.BASE}/sports/{key}/odds"
            params = {
                "apiKey": self.api_key,
                "regions": regions,
                "markets": markets,
                "oddsFormat": odds_format,
            }
            r = requests.get(url, params=params, timeout=20)
            if r.status_code == 401:
                raise RuntimeError("INVALID_KEY – check ODDS_API_KEY")
            if r.status_code == 429:
                raise RuntimeError("Rate limit / out of credits on The Odds API")
            r.raise_for_status()
            events = r.json()

            for ev in events:
                commence_raw = ev.get("commence_time")
                try:
                    commence = datetime.fromisoformat(commence_raw.replace("Z", "+00:00"))
                except Exception:
                    continue
                if commence > cutoff:
                    continue

                home = ev.get("home_team", "")
                away = ev.get("away_team", "")
                event_id = ev.get("id", f"{sport}-{home}-{away}")

                for book in ev.get("bookmakers", []):
                    book_key = book.get("key", "unknown")
                    for mkt in book.get("markets", []):
                        mkt_key = mkt.get("key", "h2h")
                        for outcome in mkt.get("outcomes", []):
                            name = outcome.get("name", "")
                            price = outcome.get("price")
                            if price is None:
                                continue
                            # The Odds API with oddsFormat=american returns American odds
                            am = float(price)
                            dec = american_to_decimal(am)
                            point = outcome.get("point")
                            sel = name
                            if point is not None:
                                sel = f"{name} {point}"

                            # crude fair_p proxy from this book's price (for model seeding)
                            fair_p = 1.0 / dec if dec > 1 else 0.5

                            out.append(MarketOdds(
                                event_id=event_id,
                                sport=sport,
                                home=home,
                                away=away,
                                commence_time=commence,
                                market=mkt_key,
                                selection=sel,
                                american_odds=am,
                                decimal_odds=dec,
                                book=book_key,
                                line=float(point) if point is not None else None,
                                implied_prob=1.0 / dec,
                                raw={"fair_p": fair_p, "source": "the-odds-api"},
                            ))

            # Prefer one book per selection to avoid explosion (DraftKings if present)
            # Keep all for now; ranking will filter by edge later.

        if not out:
            raise RuntimeError("No events returned from The Odds API for the selected sports/window")
        print(f"[OddsConnector] Live: {len(out)} market lines from The Odds API")
        return out

    def _synthetic_slate(
        self,
        sports: List[str],
        days_ahead: int,
        seed: int,
    ) -> List[MarketOdds]:
        rng = np.random.default_rng(seed)
        now = datetime.now(timezone.utc)
        teams = {
            "NFL": ["KC", "BUF", "SF", "PHI", "DAL", "MIA", "BAL", "DET"],
            "NBA": ["BOS", "DEN", "OKC", "MIN", "NYK", "MIL", "LAL", "PHX"],
            "NHL": ["COL", "EDM", "NYR", "DAL", "FLA", "VAN", "CAR", "TOR"],
            "MLB": ["LAD", "NYY", "ATL", "HOU", "PHI", "BAL", "TEX", "SD"],
            "NCAAF": ["GEO", "ALA", "OSU", "MICH", "TEX", "ORE", "PENN", "LSU"],
        }
        markets: List[MarketOdds] = []
        eid = 0
        for day in range(1, days_ahead + 1):
            commence = now + timedelta(days=day, hours=int(rng.integers(17, 22)))
            for sport in sports:
                roster = teams.get(sport, ["A", "B", "C", "D"])
                n_games = 2 if sport in ("NFL", "NCAAF") else 3
                for g in range(n_games):
                    home, away = rng.choice(roster, size=2, replace=False)
                    eid += 1
                    event_id = f"{sport}-{eid:04d}"
                    strength = rng.normal(0, 0.55)
                    p_home = 1.0 / (1.0 + np.exp(-strength))
                    p_away = 1.0 - p_home
                    juice = rng.uniform(1.04, 1.07)
                    for sel, p, team in (("home", p_home, home), ("away", p_away, away)):
                        fair_dec = 1.0 / max(p, 0.08)
                        for book, j_mult in (
                            ("pinnacle", 1.02),
                            ("draftkings", juice),
                            ("fanduel", juice * 1.01),
                            ("betmgm", juice * 1.015),
                        ):
                            misprice = 1.0
                            if book != "pinnacle" and rng.random() < 0.35:
                                misprice = float(rng.uniform(0.94, 0.99))
                            dec = max(1.12, fair_dec * (j_mult / 2 + 0.48) / misprice)
                            am = (dec - 1) * 100 if dec >= 2 else -100 / (dec - 1)
                            markets.append(MarketOdds(
                                event_id=event_id,
                                sport=sport,
                                home=str(home),
                                away=str(away),
                                commence_time=commence,
                                market="h2h",
                                selection=f"{team} ML",
                                american_odds=float(am),
                                decimal_odds=float(dec),
                                book=book,
                                raw={"fair_p": float(p), "day": day},
                            ))
        return markets
