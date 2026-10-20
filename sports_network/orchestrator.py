"""
Run the bot team — NO API KEY REQUIRED.

ESPN public + Open-Meteo + internal Elo/stats + optional Grok intel inbox.
"""

from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sports_network.storage.store import NetworkStore
from sports_network.bots.shared.weather_bot import WeatherBot
from sports_network.bots.shared.odds_bot import OddsBot
from sports_network.bots.shared.intel_bot import IntelBot
from sports_network.bots.nfl.ratings_bot import NFLRatingsBot
from sports_network.bots.nfl.injuries_bot import NFLInjuriesBot
from sports_network.bots.nfl.espn_injuries_bot import ESPNInjuriesBot
from sports_network.bots.nfl.stats_bot import NFLStatsBot
from sports_network.bots.nba.ratings_bot import NBARatingsBot
from sports_network.bots.soccer.ratings_bot import SoccerRatingsBot


BOT_TEAM = [
    OddsBot,             # ESPN public scoreboard (NFL/NBA/NHL/MLB/NCAAF)
    WeatherBot,          # Open-Meteo free
    NFLRatingsBot,       # internal Elo
    ESPNInjuriesBot,     # ESPN public injuries
    NFLInjuriesBot,      # optional env deltas
    NFLStatsBot,         # internal stats proxies
    NBARatingsBot,       # internal NBA Elo
    SoccerRatingsBot,    # soccer club Elo seeds
    IntelBot,            # Grok / manual intel inbox
]


def run_all(store: NetworkStore | None = None) -> dict:
    store = store or NetworkStore()
    results = {}
    for cls in BOT_TEAM:
        bot = cls(store)
        try:
            n = bot.run()
            results[bot.name] = {"ok": True, "n": n, "domain": bot.domain, "sport": bot.sport}
            print(f"  ✓ {bot.name:22} domain={bot.domain:10} records={n}")
        except Exception as e:
            results[bot.name] = {"ok": False, "error": str(e)}
            print(f"  ✗ {bot.name:22} ERROR {e}")
    return results


def main():
    print("=" * 64)
    print("  GROK SPORTS NETWORK — zero API key bot team")
    print("=" * 64)
    run_all()
    summary = NetworkStore().summary()
    print("\n  Inventory:")
    for row in summary.get("inventory") or []:
        print(f"    {row['domain']:10} {row['sport']:6} n={row['n']}")
    print("=" * 64)
    print("  No Odds API key needed. Optional: data/intel_inbox.json from Grok.")
    print("  API: python3 -m sports_network.api.server")
    print("  Desk: python3 desk.py")


if __name__ == "__main__":
    main()
