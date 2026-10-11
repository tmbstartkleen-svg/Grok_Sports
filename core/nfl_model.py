"""
Sharper NFL moneyline model:
  Elo + HFA + rest days + market spread prior + injury deltas.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import math
import re

DEFAULT_RATINGS: Dict[str, float] = {
    "KC": 1680, "BUF": 1650, "BAL": 1640, "DET": 1620, "PHI": 1615,
    "SF": 1610, "DAL": 1580, "MIA": 1570, "CIN": 1565, "GB": 1560,
    "HOU": 1555, "LAR": 1550, "MIN": 1545, "SEA": 1540, "TB": 1535,
    "ATL": 1520, "LAC": 1515, "PIT": 1510, "CLE": 1500, "JAX": 1495,
    "IND": 1490, "NO": 1485, "CHI": 1480, "NYJ": 1475, "LV": 1470,
    "TEN": 1465, "DEN": 1460, "ARI": 1455, "WAS": 1450, "NYG": 1445,
    "NE": 1440, "CAR": 1430,
}

ALIASES = {
    "kansas city": "KC", "chiefs": "KC", "buffalo": "BUF", "bills": "BUF",
    "baltimore": "BAL", "ravens": "BAL", "detroit": "DET", "lions": "DET",
    "philadelphia": "PHI", "eagles": "PHI", "san francisco": "SF", "49ers": "SF",
    "niners": "SF", "dallas": "DAL", "cowboys": "DAL", "miami": "MIA", "dolphins": "MIA",
    "cincinnati": "CIN", "bengals": "CIN", "green bay": "GB", "packers": "GB",
    "houston": "HOU", "texans": "HOU", "los angeles rams": "LAR", "rams": "LAR",
    "la rams": "LAR", "minnesota": "MIN", "vikings": "MIN", "seattle": "SEA",
    "seahawks": "SEA", "tampa bay": "TB", "buccaneers": "TB", "bucs": "TB",
    "atlanta": "ATL", "falcons": "ATL", "los angeles chargers": "LAC", "chargers": "LAC",
    "la chargers": "LAC", "pittsburgh": "PIT", "steelers": "PIT", "cleveland": "CLE",
    "browns": "CLE", "jacksonville": "JAX", "jaguars": "JAX", "indianapolis": "IND",
    "colts": "IND", "new orleans": "NO", "saints": "NO", "chicago": "CHI", "bears": "CHI",
    "new york jets": "NYJ", "jets": "NYJ", "las vegas": "LV", "raiders": "LV",
    "tennessee": "TEN", "titans": "TEN", "denver": "DEN", "broncos": "DEN",
    "arizona": "ARI", "cardinals": "ARI", "washington": "WAS", "commanders": "WAS",
    "new york giants": "NYG", "giants": "NYG", "new england": "NE", "patriots": "NE",
    "carolina": "CAR", "panthers": "CAR",
}

HFA_ELO = 48.0
ELO_SCALE = 400.0
K_FACTOR = 20.0
# ~25 Elo ≈ 1 point of spread (rough NFL mapping)
ELO_PER_POINT = 25.0
REST_ELO_PER_DAY = 8.0   # short week penalty / extra rest boost, capped
MAX_REST_ADJ = 24.0


def team_code(name: str) -> Optional[str]:
    if not name:
        return None
    n = name.strip()
    up = n.upper()
    if up in DEFAULT_RATINGS:
        return up
    low = n.lower()
    if low in ALIASES:
        return ALIASES[low]
    for key, code in ALIASES.items():
        if key in low:
            return code
    m = re.search(r"\b([A-Z]{2,3})\b", n)
    if m and m.group(1) in DEFAULT_RATINGS:
        return m.group(1)
    return None


def spread_to_home_win_prob(spread_home: float) -> float:
    """
    Home spread from home team perspective (negative = home favored).
    Convert to win probability via Elo-equivalent.
    e.g. -3 → home favored by 3 → +75 Elo → p > 0.5
    """
    elo_diff = (-spread_home) * ELO_PER_POINT + HFA_ELO * 0.0  # spread already embeds HFA
    # When using pure spread, market already includes HFA in the number
    return 1.0 / (1.0 + 10 ** (-elo_diff / ELO_SCALE))


@dataclass
class NFLModel:
    ratings: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_RATINGS))
    hfa: float = HFA_ELO
    # injury_delta[team_code] = Elo points (negative = hurt)
    injury_elo: Dict[str, float] = field(default_factory=dict)
    # rest_days[team_code] optional override; else assume 7
    rest_days: Dict[str, float] = field(default_factory=dict)

    def elo(self, code: str) -> float:
        base = self.ratings.get(code, 1500.0)
        inj = self.injury_elo.get(code, 0.0)
        return base + inj

    def rest_adjustment(self, home_code: str, away_code: str) -> float:
        """Positive = boost home side in Elo diff."""
        rh = self.rest_days.get(home_code, 7.0)
        ra = self.rest_days.get(away_code, 7.0)
        # relative rest: home extra days help home
        delta_days = rh - ra
        adj = max(-MAX_REST_ADJ, min(MAX_REST_ADJ, delta_days * REST_ELO_PER_DAY))
        return adj

    def win_prob(
        self,
        home: str,
        away: str,
        spread_home: Optional[float] = None,
        rest_home: Optional[float] = None,
        rest_away: Optional[float] = None,
    ) -> Tuple[float, float]:
        hc = team_code(home) or "AVG"
        ac = team_code(away) or "AVG"
        rh = self.elo(hc) if hc != "AVG" else 1500.0
        ra = self.elo(ac) if ac != "AVG" else 1500.0

        if rest_home is not None:
            self.rest_days[hc] = rest_home
        if rest_away is not None:
            self.rest_days[ac] = rest_away

        rest_adj = self.rest_adjustment(hc, ac) if hc != "AVG" and ac != "AVG" else 0.0
        diff = (rh + self.hfa + rest_adj) - ra
        p_elo = 1.0 / (1.0 + 10 ** (-diff / ELO_SCALE))

        # Blend market spread prior when available (sharp anchor)
        if spread_home is not None:
            p_spread = spread_to_home_win_prob(spread_home)
            # 50/50 Elo vs spread prior — spread is strong information
            p_home = 0.50 * p_elo + 0.50 * p_spread
        else:
            p_home = p_elo

        p_home = float(max(0.08, min(0.92, p_home)))
        return p_home, float(1.0 - p_home)

    def model_p_for_selection(
        self,
        home: str,
        away: str,
        selection: str,
        spread_home: Optional[float] = None,
        rest_home: Optional[float] = None,
        rest_away: Optional[float] = None,
    ) -> Optional[float]:
        p_home, p_away = self.win_prob(
            home, away, spread_home=spread_home, rest_home=rest_home, rest_away=rest_away
        )
        sel = (selection or "").lower()
        hc = team_code(home)
        ac = team_code(away)
        home_l = (home or "").lower()
        away_l = (away or "").lower()
        if hc and hc.lower() in sel.replace(" ", ""):
            return p_home
        if ac and ac.lower() in sel.replace(" ", ""):
            return p_away
        if home_l and any(t in sel for t in home_l.split() if len(t) > 2):
            return p_home
        if away_l and any(t in sel for t in away_l.split() if len(t) > 2):
            return p_away
        if "home" in sel:
            return p_home
        if "away" in sel:
            return p_away
        for token, p in ((home, p_home), (away, p_away)):
            if token and token.lower()[:3] in sel:
                return p
        return None

    def set_injury(self, team: str, elo_delta: float) -> None:
        code = team_code(team) or team.upper()
        self.injury_elo[code] = float(elo_delta)

    def set_rest(self, team: str, days: float) -> None:
        code = team_code(team) or team.upper()
        self.rest_days[code] = float(days)

    def update_from_result(self, home: str, away: str, home_won: bool) -> None:
        hc = team_code(home)
        ac = team_code(away)
        if not hc or not ac:
            return
        p_home, _ = self.win_prob(home, away)
        actual = 1.0 if home_won else 0.0
        rh, ra = self.ratings.get(hc, 1500.0), self.ratings.get(ac, 1500.0)
        self.ratings[hc] = rh + K_FACTOR * (actual - p_home)
        self.ratings[ac] = ra + K_FACTOR * ((1 - actual) - (1 - p_home))


_default_model: Optional[NFLModel] = None


def get_nfl_model() -> NFLModel:
    global _default_model
    if _default_model is None:
        _default_model = NFLModel()
    return _default_model
