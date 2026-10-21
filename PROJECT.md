# Project identity

**Name:** Grok Sports  
**Folder:** sports_ev_engine (legacy path; product name is Grok Sports)  
**Invoke in chat:** "Grok Sports" + task

## Live task examples

- Grok Sports — run network pull
- Grok Sports — Sunday NFL action queue
- Grok Sports — inject BUF injury -40 Elo
- Grok Sports — CLV report and gate status
- Grok Sports — add soccer bot team
- Grok Sports — path cash-out on top parlay
- Grok Sports — upgrade dashboard charts
- Grok Sports — tighten network → desk integration

## Stack

- `sports_network/` — multi-bot data supply + bridge + local API + dashboard
- `sports_network/bridge.py` — NetworkStore → MarketOdds + injury/ratings into model
- `core/` — EV, Kelly, CLV, NFL model, risk, analytics, path cash-out
- `desk.py` — CLI desk (`grok-sports>` prompt)
- `app.py` — Streamlit desk
- `requirements.txt` + `Makefile` + `start.sh` + Docker — packaging

## Bot team (current)

| Bot | Domain | Sport |
|-----|--------|-------|
| odds_bot | odds | MULTI |
| weather_bot | weather | NFL |
| nfl_ratings_bot | ratings | NFL |
| espn_injuries_bot | injuries | NFL |
| nfl_injuries_bot | injuries | NFL |
| nfl_stats_bot | stats | NFL |
| nba_ratings_bot | ratings | NBA |
| soccer_ratings_bot | ratings | SOCCER |
| mlb_ratings_bot | ratings | MLB |
| intel_bot | intel | MULTI |
