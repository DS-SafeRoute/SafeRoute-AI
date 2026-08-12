from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class QueuedObservation:
    id: int
    event_id: str
    payload: dict
    created_at: float
    attempts: int


class OfflineQueue:
    def __init__(self, db_path: str, max_age_sec: float = 86400, max_items: int = 1000) -> None:
        self.max_age_sec = max_age_sec
        self.max_items = max_items
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("CREATE TABLE IF NOT EXISTS observations (id INTEGER PRIMARY KEY AUTOINCREMENT, event_id TEXT NOT NULL UNIQUE, payload TEXT NOT NULL, created_at REAL NOT NULL, attempts INTEGER NOT NULL DEFAULT 0)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_observations_created ON observations(created_at)")
        self._conn.commit()

    def enqueue(self, event_id: str, payload: dict) -> None:
        self._conn.execute("INSERT OR IGNORE INTO observations(event_id,payload,created_at) VALUES(?,?,?)", (event_id, json.dumps(payload), time.time()))
        self._conn.commit()
        overflow = self.size() - self.max_items
        if overflow > 0:
            self._conn.execute("DELETE FROM observations WHERE id IN (SELECT id FROM observations ORDER BY created_at LIMIT ?)", (overflow,))
            self._conn.commit()

    def peek_oldest(self, limit: int = 10) -> list[QueuedObservation]:
        cutoff = time.time() - self.max_age_sec
        self._conn.execute("DELETE FROM observations WHERE created_at < ?", (cutoff,))
        self._conn.commit()
        rows = self._conn.execute("SELECT id,event_id,payload,created_at,attempts FROM observations ORDER BY created_at LIMIT ?", (limit,)).fetchall()
        return [QueuedObservation(r[0], r[1], json.loads(r[2]), r[3], r[4]) for r in rows]

    def mark_success(self, item_id: int) -> None:
        self._conn.execute("DELETE FROM observations WHERE id=?", (item_id,))
        self._conn.commit()

    def mark_failed_attempt(self, item_id: int) -> None:
        self._conn.execute("UPDATE observations SET attempts=attempts+1 WHERE id=?", (item_id,))
        self._conn.commit()

    def size(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0])

    def close(self) -> None:
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
