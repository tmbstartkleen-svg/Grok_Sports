# Grok Sports

Elite sports analysis network + EV desk.  
**Refer to this project as: `Grok Sports`**

When you want live work in chat, say for example:
- "Grok Sports — network pull and scan"
- "Grok Sports — injury intel for BUF"
- "Grok Sports — build action queue for Sunday NFL"
- "Grok Sports — CLV report and gate status"

## No API key required

Bots use ESPN public data, Open-Meteo, internal Elo, and optional Grok intel inbox.  
Optional `ODDS_API_KEY` is never required.

## Quick start

```bash
# one-time
pip install -r requirements.txt

# collect data (bot team)
python3 -m sports_network.orchestrator
# or: make network

# CLI desk
python3 desk.py
# or: make desk

# Streamlit desk
streamlit run app.py
# or: make app

# Local API + HTML dashboard
python3 -m sports_network.api.server
# open http://127.0.0.1:8787/
# or: make api
```

### CLI desk commands

```text
grok-sports> network pull
grok-sports> scan
grok-sports> q
grok-sports> report
grok-sports> history 20
grok-sports> open
grok-sports> bankroll 5000
```

### Dashboard

`http://127.0.0.1:8787/` — Network pull · Scan board · action queue · edge charts · inventory · PM sample.

## Intel from Grok chat

Save structured JSON to `data/intel_inbox.json`, then `network pull`.

## Product map

| Layer | Role |
|-------|------|
| Bot team | odds, weather, ratings, injuries, stats, soccer, intel |
| Sports Network DB | proprietary SQLite store |
| Local API | your data key for any app (`:8787`) |
| EV Desk | action queue, CLV gate, Kelly, risk, path cash-out |
| Streamlit app | full visual desk |
| CLI desk | fast terminal workflow |

## Make targets

```bash
make install   # deps
make network   # bot team
make desk      # CLI
make app       # Streamlit
make api       # dashboard + API
make clean     # pycache + local DBs
```

## Stack highlights

- **CLV gate** — Kelly / edge scale only after positive rolling CLV
- **Action queue** — empty queue is a feature (no edge → stand down)
- **Path cash-out** — Monte Carlo mid-parlay decisions
- **Risk limits** — daily / portfolio filters on the queue
- **Zero paid key** for the core loop
