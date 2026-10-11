"""
Intel inject bot — NO API KEY.

When you chat with Grok, paste structured intel into:
  data/intel_inbox.json
or set env GROK_INTEL='[{"domain":"injuries","sport":"NFL","entity_id":"BUF","payload":{...}}]'

This bot reads the inbox into the Sports Network store.
"""

from __future__ import annotations
import json
import os
from pathlib import Path
from typing import List
from sports_network.bots.base import BaseBot, BotRecord, utc_now

INBOX = Path(__file__).resolve().parent.parent.parent.parent / "data" / "intel_inbox.json"


class IntelBot(BaseBot):
    name = "intel_bot"
    domain = "intel"
    sport = "MULTI"

    def collect(self) -> List[BotRecord]:
        rows = []
        raw = os.environ.get("GROK_INTEL")
        if raw:
            try:
                rows.extend(json.loads(raw))
            except Exception as e:
                print(f"[intel_bot] GROK_INTEL parse error: {e}")
        if INBOX.exists():
            try:
                rows.extend(json.loads(INBOX.read_text()))
            except Exception as e:
                print(f"[intel_bot] inbox parse error: {e}")

        out = []
        for r in rows:
            if not isinstance(r, dict):
                continue
            domain = r.get("domain") or "intel"
            sport = r.get("sport") or "MULTI"
            eid = str(r.get("entity_id") or r.get("id") or "note")
            payload = r.get("payload") or r
            out.append(BotRecord(
                domain=domain, sport=sport, entity_id=eid,
                source=r.get("source") or "grok-intel",
                confidence=float(r.get("confidence") or 0.75),
                payload=payload if isinstance(payload, dict) else {"text": str(payload)},
                collected_at=utc_now(),
            ))
        return out
