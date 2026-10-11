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

## Stack

- sports_network/ — multi-bot data supply + local API
- core/ — EV, Kelly, CLV, NFL model, risk, analytics
- desk.py — CLI desk (prompt: grok-sports>)
- app.py — Streamlit desk
