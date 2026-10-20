.PHONY: network desk app api dashboard install clean

install:
	pip install -r requirements.txt

network:
	python3 -m sports_network.orchestrator

desk:
	python3 desk.py

app:
	streamlit run app.py

api:
	python3 -m sports_network.api.server

dashboard: api

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -f data/*.db data/sports_network.db data/bet_history.db 2>/dev/null || true

help:
	@echo "Grok Sports targets:"
	@echo "  make install   — pip install deps"
	@echo "  make network   — run bot team"
	@echo "  make desk      — CLI desk"
	@echo "  make app       — Streamlit desk"
	@echo "  make api       — local API + dashboard on :8787"
	@echo "  make clean     — remove pycache + local DBs"
