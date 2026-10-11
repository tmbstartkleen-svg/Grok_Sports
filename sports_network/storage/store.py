"""
SQLite-backed sports network store — our proprietary data layer.
"""

from __future__ import annotations
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

import os
_base = Path(__file__).resolve().parent.parent.parent / "data" / "sports_network.db"
if os.environ.get("SPORTS_NETWORK_DB"):
    DEFAULT_DB = Path(os.environ["SPORTS_NETWORK_DB"])
else:
    try:
        _base.parent.mkdir(parents=True, exist_ok=True)
        test = _base.parent / ".wtest"
        test.write_text("ok")
        test.unlink(missing_ok=True)
        DEFAULT_DB = _base
    except Exception:
        DEFAULT_DB = Path("/tmp/sports_network.db")


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class NetworkStore:
    def __init__(self, db_path: Path = DEFAULT_DB):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    @contextmanager
    def _conn(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        c = sqlite3.connect(str(self.db_path), timeout=30)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA journal_mode=WAL")
        try:
            yield c
            c.commit()
        except sqlite3.OperationalError:
            try:
                c.rollback()
            except Exception:
                pass
            raise
        finally:
            c.close()

    def _init(self):
        with self._conn() as c:
            c.execute("""
            CREATE TABLE IF NOT EXISTS records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                domain TEXT NOT NULL,
                sport TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                source TEXT,
                confidence REAL,
                collected_at TEXT,
                payload_json TEXT,
                UNIQUE(domain, sport, entity_id, source)
            )""")
            c.execute("""
            CREATE TABLE IF NOT EXISTS bot_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                bot_name TEXT,
                domain TEXT,
                sport TEXT,
                n_records INTEGER,
                ran_at TEXT
            )""")
            c.execute("CREATE INDEX IF NOT EXISTS idx_domain_sport ON records(domain, sport)")

    def write(self, record) -> None:
        d = record.to_dict() if hasattr(record, "to_dict") else record
        with self._conn() as c:
            c.execute("""
            INSERT INTO records (domain, sport, entity_id, source, confidence, collected_at, payload_json)
            VALUES (?,?,?,?,?,?,?)
            ON CONFLICT(domain, sport, entity_id, source) DO UPDATE SET
                confidence=excluded.confidence,
                collected_at=excluded.collected_at,
                payload_json=excluded.payload_json
            """, (
                d["domain"], d["sport"], d["entity_id"], d.get("source"),
                d.get("confidence", 1.0), d.get("collected_at", _utc()),
                json.dumps(d.get("payload") or {}),
            ))

    def log_run(self, bot_name: str, domain: str, sport: str, n: int) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO bot_runs (bot_name, domain, sport, n_records, ran_at) VALUES (?,?,?,?,?)",
                (bot_name, domain, sport, n, _utc()),
            )

    def query(
        self,
        domain: Optional[str] = None,
        sport: Optional[str] = None,
        entity_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[dict]:
        clauses, args = [], []
        if domain:
            clauses.append("domain=?"); args.append(domain)
        if sport:
            clauses.append("sport=?"); args.append(sport)
        if entity_id:
            clauses.append("entity_id=?"); args.append(entity_id)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"SELECT * FROM records {where} ORDER BY collected_at DESC LIMIT ?"
        args.append(limit)
        with self._conn() as c:
            rows = c.execute(sql, args).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["payload"] = json.loads(d.pop("payload_json") or "{}")
            out.append(d)
        return out

    def latest_by_domain(self, domain: str, sport: Optional[str] = None) -> List[dict]:
        return self.query(domain=domain, sport=sport, limit=200)

    def summary(self) -> dict:
        with self._conn() as c:
            by_domain = c.execute(
                "SELECT domain, sport, COUNT(*) n FROM records GROUP BY domain, sport"
            ).fetchall()
            runs = c.execute(
                "SELECT bot_name, domain, n_records, ran_at FROM bot_runs ORDER BY id DESC LIMIT 20"
            ).fetchall()
        return {
            "inventory": [dict(r) for r in by_domain],
            "recent_runs": [dict(r) for r in runs],
        }
