#!/usr/bin/env bash
# One-command Grok Sports launcher
set -e
cd "$(dirname "$0")"

MODE="${1:-api}"

if [[ ! -d .venv ]] && command -v python3 >/dev/null; then
  echo "[start] creating .venv…"
  python3 -m venv .venv
  .venv/bin/pip install -q -r requirements.txt
fi

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

case "$MODE" in
  network|collect)
    exec python3 -m sports_network.orchestrator
    ;;
  desk)
    exec python3 desk.py
    ;;
  app|streamlit)
    exec streamlit run app.py --server.headless true
    ;;
  api|dashboard|*)
    echo "[start] API + dashboard → http://127.0.0.1:${NETWORK_PORT:-8787}/"
    exec python3 -m sports_network.api.server
    ;;
esac
