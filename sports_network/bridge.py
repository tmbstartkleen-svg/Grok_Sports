"""
Bridge: NetworkStore → MarketOdds + injury/ratings deltas for the edge model.

Turns proprietary bot data into the same MarketOdds objects the EV desk already
consumes, so network pull directly feeds scan / action queue.
"""

from __future__ import annotations
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import re

from data.odds_connector import MarketOdds, american_to_decimal
from sports_network.storage.store import NetworkStore


def _parse_commence(raw: Any) -> datetime:
    if isinstance(raw, datetime):
        return raw if raw.tzinfo else raw.replace(tzinfo=timezone.utc)
    if not raw:
        return datetime.now(timezone.utc)
    s = str(raw).replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return datetime.now(timezone.utc)


def _ml_to_american(ml) -> Optional[float]:
    if ml is None:
        return None
    try:
        return float(ml)
    except Exception:
        return None


def network_odds_to_markets(store: Optional[NetworkStore] = None, limit: int = 120) -> List[MarketOdds]:
    """Convert domain=odds records into MarketOdds lines (h2h)."""
    store = store or NetworkStore()
    rows = store.query(domain="odds", limit=limit)
    out: List[MarketOdds] = []
    for r in rows:
        p = r.get("payload") or {}
        sport = (r.get("sport") or "MULTI").upper()
        home = p.get("home") or p.get("home_abbr") or "?"
        away = p.get("away") or p.get("away_abbr") or "?"
        eid = str(r.get("entity_id") or f"{sport}-{home}-{away}")
        commence = _parse_commence(p.get("commence"))
        book = (r.get("source") or "network").replace("-public", "").replace("-", "")[:16] or "network"

        for side, team, ml_key in (
            ("home", home, "home_ml"),
            ("away", away, "away_ml"),
        ):
            am = _ml_to_american(p.get(ml_key))
            if am is None:
                continue
            dec = american_to_decimal(am)
            out.append(MarketOdds(
                event_id=eid,
                sport=sport,
                home=str(home),
                away=str(away),
                commence_time=commence,
                market="h2h",
                selection=f"{team} ML",
                american_odds=float(am),
                decimal_odds=float(dec),
                book=book,
                raw={
                    "fair_p": 1.0 / dec if dec > 1 else 0.5,
                    "source": "sports-network",
                    "spread_home": p.get("spread_home"),
                    "total": p.get("total"),
                    "status": p.get("status"),
                },
            ))

        spread = p.get("spread_home")
        if spread is not None:
            try:
                sp = float(spread)
                for sel, am in ((f"{home} {sp:+.1f}", -110.0), (f"{away} {-sp:+.1f}", -110.0)):
                    dec = american_to_decimal(am)
                    out.append(MarketOdds(
                        event_id=eid,
                        sport=sport,
                        home=str(home),
                        away=str(away),
                        commence_time=commence,
                        market="spreads",
                        selection=sel,
                        american_odds=am,
                        decimal_odds=dec,
                        book=book,
                        line=sp if "home" in sel.lower() or home.lower()[:3] in sel.lower() else -sp,
                        raw={"source": "sports-network", "spread_home": sp},
                    ))
            except Exception:
                pass
    return out


def network_injury_deltas(store: Optional[NetworkStore] = None) -> Dict[str, float]:
    store = store or NetworkStore()
    rows = store.query(domain="injuries", limit=200)
    deltas: Dict[str, float] = {}
    for r in rows:
        p = r.get("payload") or {}
        team = (p.get("team") or p.get("team_abbr") or r.get("entity_id") or "").upper()
        team = re.sub(r"[^A-Z]", "", team)[:3]
        if not team:
            continue
        if p.get("elo_delta") is not None:
            try:
                deltas[team] = deltas.get(team, 0.0) + float(p["elo_delta"])
                continue
            except Exception:
                pass
        status = str(p.get("status") or p.get("injury_status") or "").lower()
        if any(x in status for x in ("out", "ir", "doubtful")):
            deltas[team] = deltas.get(team, 0.0) - 25.0
        elif "questionable" in status:
            deltas[team] = deltas.get(team, 0.0) - 10.0
    return deltas


def network_ratings_map(store: Optional[NetworkStore] = None, sport: str = "NFL") -> Dict[str, float]:
    store = store or NetworkStore()
    rows = store.query(domain="ratings", sport=sport, limit=64)
    out = {}
    for r in rows:
        p = r.get("payload") or {}
        team = (p.get("team") or p.get("club") or r.get("entity_id") or "").upper()
        elo = p.get("effective_elo") or p.get("elo")
        if team and elo is not None:
            try:
                out[team] = float(elo)
            except Exception:
                pass
    return out


def enrich_nfl_model_from_network(store: Optional[NetworkStore] = None) -> int:
    try:
        from core.nfl_model import get_nfl_model
    except Exception:
        return 0
    store = store or NetworkStore()
    model = get_nfl_model()
    n = 0
    for team, elo in network_ratings_map(store, "NFL").items():
        model.ratings[team] = elo
        n += 1
    for team, delta in network_injury_deltas(store).items():
        model.set_injury(team, delta)
        n += 1
    return n


def fetch_slate_with_network(
    sports: Optional[List[str]] = None,
    days_ahead: int = 5,
    seed: int = 42,
    prefer_network: bool = True,
) -> tuple:
    from data.odds_connector import OddsConnector

    store = NetworkStore()
    notes = []
    inj = network_injury_deltas(store)
    n_enrich = enrich_nfl_model_from_network(store)
    if n_enrich:
        notes.append(f"nfl-model-enriched:{n_enrich}")
    if inj:
        notes.append(f"inj-deltas:{len(inj)}")

    net_markets = network_odds_to_markets(store) if prefer_network else []
    if net_markets:
        notes.append(f"network-odds:{len(net_markets)}")
        try:
            extra = OddsConnector().fetch_slate(sports=sports, days_ahead=days_ahead, seed=seed)
            notes.append(f"sports-extra:{len(extra)}")
            net_ids = {m.event_id for m in net_markets}
            merged = list(net_markets) + [m for m in extra if m.event_id not in net_ids]
            return merged, notes, inj
        except Exception:
            return net_markets, notes, inj

    try:
        markets = OddsConnector().fetch_slate(sports=sports, days_ahead=days_ahead, seed=seed)
        notes.append(f"sports:{'LIVE' if OddsConnector().api_key else 'synth'}:{len(markets)}")
    except Exception:
        markets = OddsConnector(api_key=None)._synthetic_slate(
            sports or ["NFL", "NCAAF", "NBA", "NHL"], days_ahead, seed
        )
        notes.append("sports:fallback")
    return markets, notes, inj
