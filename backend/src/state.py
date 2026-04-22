"""
Shared in-memory state between the pipeline and the web dashboard.
Thread-safe for pipeline writes; async-safe for the web server reads.
"""

import asyncio
import json
import logging
import sqlite3
import sys
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

STEPS = [
    "queued",
    "detected",
    "fetching",
    "forking",
    "cloning",
    "setup",
    "solving",
    "pushing",
    "done",
]

# Logs persist in a SQLite file so the dashboard can query ranges across restarts.
# Retention matches the longest dropdown range — anything older is pruned at startup.
_LOG_DB_PATH = Path(__file__).resolve().parent.parent / "logs.db"
LOG_RETENTION_DAYS = 3

_issues: dict[str, dict] = {}
_websockets: set = set()
_main_loop: asyncio.AbstractEventLoop | None = None
_log_queue: asyncio.Queue | None = None
_db: sqlite3.Connection | None = None
_db_lock = threading.Lock()


def init(loop: asyncio.AbstractEventLoop) -> None:
    global _main_loop, _log_queue
    _main_loop = loop
    _log_queue = asyncio.Queue()
    _init_db()


def _init_db() -> None:
    global _db
    _db = sqlite3.connect(_LOG_DB_PATH, check_same_thread=False, isolation_level=None)
    _db.execute(
        "CREATE TABLE IF NOT EXISTS logs ("
        "ts TEXT NOT NULL, issue_id TEXT, message TEXT NOT NULL)"
    )
    _db.execute("CREATE INDEX IF NOT EXISTS idx_logs_ts ON logs(ts)")
    cutoff = (datetime.now(timezone.utc) - timedelta(days=LOG_RETENTION_DAYS)).isoformat()
    _db.execute("DELETE FROM logs WHERE ts < ?", (cutoff,))


# ------------------------------------------------------------------ #
# Issue state                                                          #
# ------------------------------------------------------------------ #

def upsert_issue(issue_id: str, **kwargs) -> None:
    if issue_id not in _issues:
        _issues[issue_id] = {
            "id": issue_id,
            "title": issue_id,
            "step": "detected",
            "failed": False,
            "pr_url": None,
            "error": None,
            "started_at": _now(),
            "updated_at": _now(),
        }
    _issues[issue_id].update(kwargs)
    _issues[issue_id]["updated_at"] = _now()
    _push_event({"type": "issue_update", "issue": _issues[issue_id]})


def get_all() -> list[dict]:
    return list(_issues.values())


# ------------------------------------------------------------------ #
# Logging                                                              #
# ------------------------------------------------------------------ #

def log(issue_id: str | None, message: str) -> None:
    ts = _now()
    _persist_log(ts, issue_id, message)
    _push_event({"type": "log", "issue_id": issue_id, "message": message, "ts": ts})


def _persist_log(ts: str, issue_id: str | None, message: str) -> None:
    if _db is None:
        return
    try:
        with _db_lock:
            _db.execute(
                "INSERT INTO logs (ts, issue_id, message) VALUES (?, ?, ?)",
                (ts, issue_id, message),
            )
    except Exception as e:
        # Write directly to stderr instead of using the `logging` module —
        # the root logger has StateLogHandler attached, which calls log() →
        # _persist_log, producing unbounded recursion on persistent failures.
        print(f"[state] Failed to persist log ({ts}): {e}", file=sys.stderr)


def get_logs_since(since_iso: str) -> list[dict]:
    if _db is None:
        return []
    with _db_lock:
        rows = _db.execute(
            "SELECT ts, issue_id, message FROM logs WHERE ts >= ? ORDER BY ts ASC",
            (since_iso,),
        ).fetchall()
    return [{"ts": r[0], "issue_id": r[1], "message": r[2]} for r in rows]


# ------------------------------------------------------------------ #
# WebSocket fan-out                                                    #
# ------------------------------------------------------------------ #

def register_ws(ws) -> None:
    _websockets.add(ws)


def unregister_ws(ws) -> None:
    _websockets.discard(ws)


async def broadcaster() -> None:
    """Async task: drains the log queue and fans out to all WebSocket clients."""
    while True:
        item = await _log_queue.get()
        payload = json.dumps(item)
        dead = set()
        for ws in list(_websockets):
            try:
                await ws.send_text(payload)
            except Exception:
                dead.add(ws)
        _websockets.difference_update(dead)


# ------------------------------------------------------------------ #
# Internal helpers                                                     #
# ------------------------------------------------------------------ #

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _push_event(event: dict) -> None:
    if _main_loop and _log_queue:
        _main_loop.call_soon_threadsafe(_log_queue.put_nowait, event)


# ------------------------------------------------------------------ #
# Logging handler — routes Python log records into the state system   #
# ------------------------------------------------------------------ #

class StateLogHandler(logging.Handler):
    def __init__(self, issue_id_getter=None):
        super().__init__()
        self._getter = issue_id_getter

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            issue_id = self._getter() if self._getter else None
            log(issue_id, msg)
        except Exception:
            pass
