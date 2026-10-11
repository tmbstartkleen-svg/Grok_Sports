"""
Prediction markets connector (Polymarket Gamma API – no key required).

Fetches active binary markets, maps YES/NO prices to decimal odds + model legs.
Also ships a synthetic PM slate for offline demos.
"""

from __future__ import annotations
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict, Any
import numpy as np

try:
    import requests
except ImportError:
    requests = None

GAMMA = "https://gamma-api.polymarket.com"


@dataclass
class PredMarket:
    market_id: str
    question: str
    outcome: str                 # "Yes" or "No"
    price: float                 # 0-1 share price
    decimal_odds: float          # 1/price
    volume: float = 0.0
    liquidity: float = 0.0
    end_date: Optional[str] = None
    category: str = "prediction"
    event_slug: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def implied_prob(self) -> float:
        return float(np.clip(self.price, 0.01, 0.99))


class PredictionMarketConnector:
    """Polymarket Gamma + synthetic fallback."""

    def __init__(self, use_live: bool = True):
        self.use_live = use_live and requests is not None

    def fetch_markets(
        self,
        limit: int = 80,
        min_volume: float = 1000.0,
        seed: int = 42,
        sources: Optional[List[str]] = None,
    ) -> List[PredMarket]:
        sources = sources or ["polymarket", "kalshi"]
        if not self.use_live:
            return self._synthetic(limit=min(limit, 40), seed=seed)
        out: List[PredMarket] = []
        if "polymarket" in sources:
            try:
                out.extend(self._fetch_polymarket(limit=limit, min_volume=min_volume))
            except Exception as e:
                print(f"[PM] Polymarket failed ({e})")
        if "kalshi" in sources:
            try:
                out.extend(self._fetch_kalshi(limit=limit))
            except Exception as e:
                print(f"[PM] Kalshi failed ({e})")
        if not out:
            print("[PM] All live sources failed; synthetic.")
            return self._synthetic(limit=min(limit, 40), seed=seed)
        return out

    def _fetch_polymarket(self, limit: int, min_volume: float) -> List[PredMarket]:
        assert requests is not None
        out: List[PredMarket] = []
        # High-volume active events
        r = requests.get(
            f"{GAMMA}/events",
            params={
                "active": "true",
                "closed": "false",
                "order": "volume24hr",
                "ascending": "false",
                "limit": min(limit, 50),
            },
            timeout=20,
        )
        r.raise_for_status()
        events = r.json()
        for ev in events:
            title = ev.get("title") or ev.get("slug") or ""
            slug = ev.get("slug") or ""
            for m in ev.get("markets") or []:
                if m.get("closed"):
                    continue
                try:
                    outcomes = m.get("outcomes")
                    prices = m.get("outcomePrices")
                    if isinstance(outcomes, str):
                        outcomes = json.loads(outcomes)
                    if isinstance(prices, str):
                        prices = json.loads(prices)
                    if not outcomes or not prices or len(outcomes) != len(prices):
                        continue
                    vol = float(m.get("volumeNum") or m.get("volume") or 0)
                    liq = float(m.get("liquidityNum") or m.get("liquidity") or 0)
                    if vol < min_volume and liq < min_volume * 0.5:
                        continue
                    mid = str(m.get("id") or m.get("conditionId") or "")
                    q = m.get("question") or title
                    end = m.get("endDate") or m.get("end_date_iso") or ev.get("endDate")
                    for outcome, price_s in zip(outcomes, prices):
                        price = float(price_s)
                        if price < 0.03 or price > 0.97:
                            continue  # near-certain / dead
                        dec = 1.0 / price
                        out.append(PredMarket(
                            market_id=f"{mid}|{outcome}",
                            question=q[:120],
                            outcome=str(outcome),
                            price=price,
                            decimal_odds=dec,
                            volume=vol,
                            liquidity=liq,
                            end_date=str(end)[:10] if end else None,
                            category="polymarket",
                            event_slug=slug,
                            raw={"event": title},
                        ))
                except Exception:
                    continue
                if len(out) >= limit * 2:
                    break
            if len(out) >= limit * 2:
                break
        print(f"[PM] Polymarket: {len(out)} outcome lines")
        return out[: limit * 2]



    def _fetch_kalshi(self, limit: int = 40) -> List[PredMarket]:
        assert requests is not None
        base = "https://api.elections.kalshi.com/trade-api/v2"
        out: List[PredMarket] = []
        r = requests.get(
            f"{base}/markets",
            params={"limit": min(200, max(limit * 2, 50)), "status": "open"},
            timeout=25,
        )
        r.raise_for_status()
        markets = (r.json() or {}).get("markets") or []
        for m in markets:
            title = m.get("title") or m.get("ticker") or ""
            ticker = m.get("ticker") or ""
            def _f(x):
                try:
                    return float(x) if x is not None else None
                except Exception:
                    return None
            yes_ask = _f(m.get("yes_ask_dollars") or m.get("yes_ask"))
            yes_bid = _f(m.get("yes_bid_dollars") or m.get("yes_bid"))
            last = _f(m.get("last_price_dollars") or m.get("last_price"))
            yes_price = yes_ask or last or yes_bid
            if yes_price is None or yes_price <= 0.02 or yes_price >= 0.98:
                continue
            vol = _f(m.get("volume_fp") or m.get("volume")) or 0.0
            end = m.get("close_time") or m.get("expiration_time")
            for outcome, price in (("Yes", yes_price), ("No", 1.0 - yes_price)):
                if price < 0.03 or price > 0.97:
                    continue
                out.append(PredMarket(
                    market_id=f"kalshi:{ticker}|{outcome}",
                    question=title[:120],
                    outcome=outcome,
                    price=price,
                    decimal_odds=1.0 / price,
                    volume=vol,
                    liquidity=0.0,
                    end_date=str(end)[:10] if end else None,
                    category="kalshi",
                    event_slug=m.get("event_ticker") or ticker,
                    raw={"ticker": ticker},
                ))
        print(f"[PM] Kalshi: {len(out)} outcome lines")
        return out[: limit * 2]

    def _synthetic(self, limit: int, seed: int) -> List[PredMarket]:
        rng = np.random.default_rng(seed)
        topics = [
            "Fed rate cut by December?",
            "BTC above 150k before 2027?",
            "US recession in 2027?",
            "AI bill passes Senate 2026?",
            "SpaceX Starship orbital success Q4?",
            "Oil above $100 by year end?",
            "Major hurricane hits FL this season?",
            "Championship favorite wins title?",
            "Unemployment below 4% next print?",
            "ETH ETF inflows positive this month?",
        ]
        out = []
        for i, q in enumerate(topics[: max(5, limit // 2)]):
            p_yes = float(np.clip(rng.uniform(0.22, 0.78), 0.05, 0.95))
            # slight mispricing vs "true"
            true = float(np.clip(p_yes + rng.uniform(-0.04, 0.05), 0.1, 0.9))
            for outcome, price, fair in (
                ("Yes", p_yes, true),
                ("No", 1 - p_yes, 1 - true),
            ):
                # juice on price
                priced = float(np.clip(price * rng.uniform(1.0, 1.04), 0.04, 0.96))
                out.append(PredMarket(
                    market_id=f"synth-pm-{i}|{outcome}",
                    question=q,
                    outcome=outcome,
                    price=priced,
                    decimal_odds=1.0 / priced,
                    volume=float(rng.uniform(5e3, 5e5)),
                    liquidity=float(rng.uniform(1e3, 5e4)),
                    end_date=(datetime.now(timezone.utc) + timedelta(days=int(rng.integers(3, 60)))).strftime("%Y-%m-%d"),
                    category="synthetic_pm",
                    event_slug=f"synth-{i}",
                    raw={"fair_p": fair},
                ))
        return out
