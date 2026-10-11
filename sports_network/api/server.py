"""
Local Grok Sports Network API — YOU are the data key.

  python3 -m sports_network.api.server

Endpoints (no external API key required for consumers of *this* service):
  GET /health
  GET /summary
  GET /records?domain=ratings&sport=NFL
  GET /ratings/NFL
  GET /weather/NFL
  GET /odds?sport=NFL
  POST /collect   — trigger bot team run
"""

from __future__ import annotations
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sports_network.storage.store import NetworkStore
from sports_network.orchestrator import run_all


class Handler(BaseHTTPRequestHandler):
    store = NetworkStore()

    def log_message(self, fmt, *args):
        print(f"[api] {args[0]}")

    def _json(self, code: int, obj):
        body = json.dumps(obj, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        path = u.path.rstrip("/") or "/"
        qs = parse_qs(u.query)

        if path == "/health":
            return self._json(200, {"status": "ok", "network": "grok-sports-network"})

        if path == "/summary":
            return self._json(200, self.store.summary())

        if path == "/records":
            domain = (qs.get("domain") or [None])[0]
            sport = (qs.get("sport") or [None])[0]
            limit = int((qs.get("limit") or ["100"])[0])
            return self._json(200, {
                "records": self.store.query(domain=domain, sport=sport, limit=limit)
            })

        if path.startswith("/ratings/"):
            sport = path.split("/")[-1].upper()
            return self._json(200, {"sport": sport, "ratings": self.store.query("ratings", sport)})

        if path.startswith("/weather/"):
            sport = path.split("/")[-1].upper()
            return self._json(200, {"sport": sport, "weather": self.store.query("weather", sport)})

        if path == "/odds":
            sport = (qs.get("sport") or [None])[0]
            return self._json(200, {"odds": self.store.query("odds", sport, limit=50)})

        if path == "/injuries":
            return self._json(200, {"injuries": self.store.query("injuries", "NFL")})

        if path == "/stats":
            sport = (qs.get("sport") or ["NFL"])[0]
            return self._json(200, {"stats": self.store.query("stats", sport)})

        return self._json(404, {"error": "not found", "paths": [
            "/health", "/summary", "/records", "/ratings/NFL", "/weather/NFL",
            "/odds", "/injuries", "/stats", "POST /collect",
        ]})

    def do_POST(self):
        u = urlparse(self.path)
        if u.path.rstrip("/") == "/collect":
            results = run_all(self.store)
            return self._json(200, {"collected": results, "summary": self.store.summary()})
        return self._json(404, {"error": "not found"})


def main():
    port = int(__import__("os").environ.get("NETWORK_PORT", "8787"))
    httpd = HTTPServer(("0.0.0.0", port), Handler)
    print(f"Grok Sports Network API on http://127.0.0.1:{port}")
    print("  GET  /health /summary /ratings/NFL /weather/NFL /odds /injuries")
    print("  POST /collect")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
