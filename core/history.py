"""
Bet history + Closing Line Value (CLV) tracker.

SQLite store under data/bet_history.db
CLV = your taken implied prob vs closing implied prob (positive = you beat the close).
"""

from __future__ import annotations
import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Dict, Any
import numpy as np

DEFAULT_DB = Path(__file__).resolve().parent.parent / "data" / "bet_history.db"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


@contextmanager
def _conn(db_path: Path):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(db_path))
    c.row_factory = sqlite3.Row
    try:
        yield c
        c.commit()
    finally:
        c.close()


def init_db(db_path: Path = DEFAULT_DB) -> None:
    with _conn(db_path) as c:
        c.execute("""
        CREATE TABLE IF NOT EXISTS bets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            ticket_id TEXT,
            description TEXT,
            n_legs INTEGER,
            source TEXT,
            model_p REAL,
            decimal_odds REAL,
            american_odds REAL,
            edge REAL,
            stake REAL,
            book TEXT,
            status TEXT DEFAULT 'open',
            settled_at TEXT,
            won INTEGER,
            pnl REAL,
            close_decimal REAL,
            close_implied REAL,
            clv REAL,
            notes TEXT,
            raw_json TEXT,
            board_decimal REAL,
            executed_decimal REAL,
            execution_slippage REAL
        )""")
        c.execute("""
        CREATE TABLE IF NOT EXISTS open_positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            bet_id INTEGER,
            legs_json TEXT,
            legs_resolved INTEGER DEFAULT 0,
            remaining_json TEXT,
            last_offer REAL,
            last_ride_ev REAL,
            last_action TEXT,
            updated_at TEXT,
            FOREIGN KEY(bet_id) REFERENCES bets(id)
        )""")
        # migrate older DBs
        cols = {r[1] for r in c.execute("PRAGMA table_info(bets)").fetchall()}
        for col, decl in (
            ("board_decimal", "REAL"),
            ("executed_decimal", "REAL"),
            ("execution_slippage", "REAL"),
        ):
            if col not in cols:
                c.execute(f"ALTER TABLE bets ADD COLUMN {col} {decl}")


@dataclass
class BetRecord:
    ticket_id: str
    description: str
    n_legs: int
    source: str
    model_p: float
    decimal_odds: float
    stake: float
    book: str = ""
    edge: float = 0.0
    american_odds: float = 0.0
    notes: str = ""
    raw: Optional[dict] = None
    board_decimal: Optional[float] = None   # odds when scanned
    executed_decimal: Optional[float] = None  # odds actually taken


class BetHistory:
    def __init__(self, db_path: Path = DEFAULT_DB):
        self.db_path = Path(db_path)
        init_db(self.db_path)

    def log_bet(self, bet: BetRecord, legs: Optional[list] = None) -> int:
        board = bet.board_decimal if bet.board_decimal is not None else bet.decimal_odds
        executed = bet.executed_decimal if bet.executed_decimal is not None else bet.decimal_odds
        slip = None
        if board and executed and board > 1:
            # negative slippage = worse price than board
            slip = float(executed - board)
        with _conn(self.db_path) as c:
            cur = c.execute(
                """INSERT INTO bets
                (created_at, ticket_id, description, n_legs, source, model_p,
                 decimal_odds, american_odds, edge, stake, book, status, notes, raw_json,
                 board_decimal, executed_decimal, execution_slippage)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,'open',?,?,?,?,?)""",
                (
                    _utc_now(), bet.ticket_id, bet.description, bet.n_legs, bet.source,
                    bet.model_p, executed, bet.american_odds, bet.edge,
                    bet.stake, bet.book, bet.notes,
                    json.dumps(bet.raw or {}),
                    board, executed, slip,
                ),
            )
            bet_id = int(cur.lastrowid)
            if legs is not None:
                c.execute(
                    """INSERT INTO open_positions
                    (bet_id, legs_json, legs_resolved, remaining_json, updated_at)
                    VALUES (?,?,0,?,?)""",
                    (bet_id, json.dumps(legs), json.dumps(legs), _utc_now()),
                )
            return bet_id

    def list_open(self) -> List[dict]:
        with _conn(self.db_path) as c:
            rows = c.execute(
                """SELECT b.*, o.legs_resolved, o.remaining_json, o.last_offer, o.last_ride_ev, o.last_action
                   FROM bets b
                   LEFT JOIN open_positions o ON o.bet_id = b.id
                   WHERE b.status = 'open'
                   ORDER BY b.id DESC"""
            ).fetchall()
            return [dict(r) for r in rows]

    def list_recent(self, limit: int = 50) -> List[dict]:
        with _conn(self.db_path) as c:
            rows = c.execute(
                "SELECT * FROM bets ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]

    def settle(self, bet_id: int, won: bool, pnl: float, close_decimal: Optional[float] = None) -> None:
        close_imp = (1.0 / close_decimal) if close_decimal and close_decimal > 1 else None
        with _conn(self.db_path) as c:
            row = c.execute("SELECT decimal_odds FROM bets WHERE id=?", (bet_id,)).fetchone()
            clv = None
            if row and close_imp is not None:
                taken_imp = 1.0 / float(row["decimal_odds"]) if row["decimal_odds"] else None
                if taken_imp is not None:
                    # positive CLV: you got a better price than close (lower implied for a back)
                    clv = float(close_imp - taken_imp)
            c.execute(
                """UPDATE bets SET status=?, settled_at=?, won=?, pnl=?,
                   close_decimal=?, close_implied=?, clv=? WHERE id=?""",
                (
                    "won" if won else "lost",
                    _utc_now(),
                    1 if won else 0,
                    pnl,
                    close_decimal,
                    close_imp,
                    clv,
                    bet_id,
                ),
            )

    def cashout(self, bet_id: int, cashout_amount: float, stake: float) -> None:
        pnl = cashout_amount - stake
        with _conn(self.db_path) as c:
            c.execute(
                """UPDATE bets SET status='cashed', settled_at=?, won=?, pnl=? WHERE id=?""",
                (_utc_now(), 1 if pnl >= 0 else 0, pnl, bet_id),
            )

    def update_path(self, bet_id: int, remaining: list, offer: float, ride_ev: float, action: str) -> None:
        with _conn(self.db_path) as c:
            c.execute(
                """UPDATE open_positions SET remaining_json=?, last_offer=?, last_ride_ev=?,
                   last_action=?, updated_at=?, legs_resolved = legs_resolved + 1
                   WHERE bet_id=?""",
                (json.dumps(remaining), offer, ride_ev, action, _utc_now(), bet_id),
            )

    def set_executed(self, bet_id: int, executed_decimal: float) -> None:
        """Update the price you actually got (may differ from board)."""
        with _conn(self.db_path) as c:
            row = c.execute(
                "SELECT board_decimal, decimal_odds FROM bets WHERE id=?", (bet_id,)
            ).fetchone()
            board = (row["board_decimal"] if row and row["board_decimal"] else
                     (row["decimal_odds"] if row else None))
            slip = (float(executed_decimal) - float(board)) if board else None
            c.execute(
                """UPDATE bets SET executed_decimal=?, decimal_odds=?, execution_slippage=?
                   WHERE id=?""",
                (executed_decimal, executed_decimal, slip, bet_id),
            )

    def mark_leg_lost(self, bet_id: int, stake: float) -> None:

        self.settle(bet_id, won=False, pnl=-stake)

    def stats(self) -> Dict[str, Any]:
        with _conn(self.db_path) as c:
            rows = c.execute(
                "SELECT * FROM bets WHERE status IN ('won','lost','cashed')"
            ).fetchall()
        if not rows:
            return {
                "n": 0, "roi": 0.0, "pnl": 0.0, "win_rate": 0.0,
                "avg_clv": None, "clv_n": 0, "stake_total": 0.0,
            }
        pnls = [float(r["pnl"] or 0) for r in rows]
        stakes = [float(r["stake"] or 0) for r in rows]
        wins = [r for r in rows if r["won"] == 1]
        clvs = [float(r["clv"]) for r in rows if r["clv"] is not None]
        stake_total = sum(stakes) or 1.0
        slips = []
        for r in rows:
            try:
                if r["execution_slippage"] is not None:
                    slips.append(float(r["execution_slippage"]))
            except Exception:
                pass
        return {
            "n": len(rows),
            "pnl": float(sum(pnls)),
            "roi": float(sum(pnls) / stake_total),
            "win_rate": float(len(wins) / len(rows)),
            "avg_clv": float(np.mean(clvs)) if clvs else None,
            "clv_n": len(clvs),
            "stake_total": float(sum(stakes)),
            "positive_clv_rate": float(np.mean([c > 0 for c in clvs])) if clvs else None,
            "avg_slippage": float(np.mean(slips)) if slips else None,
        }
