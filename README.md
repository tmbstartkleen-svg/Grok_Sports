# Grok Sports

Elite sports analysis network + EV desk.  
**Refer to this project as: `Grok Sports`**

When you want live work in chat, say for example:
- "Grok Sports — network pull and scan"
- "Grok Sports — injury intel for BUF"
- "Grok Sports — build action queue for Sunday NFL"

## No API key required

Bots use ESPN public data, Open-Meteo, internal Elo, and optional Grok intel inbox.

## Quick start

```bash
cd sports_ev_engine   # or your Grok Sports folder

python3 -m sports_network.orchestrator   # bot team collect
python3 desk.py                          # trading desk
```

```text
desk> network pull
desk> scan
desk> q
desk> report
```

Optional local API (you are the key):

```bash
python3 -m sports_network.api.server
# http://127.0.0.1:8787
```

## Intel from Grok chat

Save structured JSON to `data/intel_inbox.json`, then `network pull`.

## Product map

| Layer | Role |
|-------|------|
| Bot team | odds, weather, ratings, injuries, stats, intel |
| Sports Network DB | proprietary store |
| Local API | your data key for any app |
| EV Desk | action queue, CLV gate, Kelly, risk, path cash-out |
