# Elite map vs market leaders

| Competitor capability | Our module |
|----------------------|------------|
| Pickswise curated picks | Action queue + min edge filters |
| Action Network tracking | Bet history + CLV + executed odds |
| TeamRankings ratings | NFL/NBA ratings bots + Elo model |
| Forebet probabilities | Edge model + Monte Carlo |
| OddsJam EV + CLV | EV engine + CLV gate + segmented report |
| Unabated custom EV | Factors + NFL spread/rest/injury + portfolio |
| Dimers simulations | monte_carlo + path cash-out |

## Bot team (data supply)

| Bot | Domain | Sport |
|-----|--------|-------|
| odds_bot | odds | MULTI |
| weather_bot | weather | NFL |
| nfl_ratings_bot | ratings | NFL |
| nfl_injuries_bot | injuries | NFL |
| nfl_stats_bot | stats | NFL |
| nba_ratings_bot | ratings | NBA |
| soccer_ratings_bot | ratings | SOCCER |

Add soccer/MLB bots the same way: one domain per bot, write to NetworkStore.

## You are the API

`python3 -m sports_network.api.server` → localhost:8787  
Consumers (desk, app, future apps) use **SPORTS_NETWORK_URL** — not scattered third-party keys.
Odds API key is only an *upstream* for odds_bot; everything else is our network.
