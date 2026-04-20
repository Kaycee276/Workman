"""
Shared in-memory state between the pipeline and the web dashboard.
Thread-safe for pipeline writes; async-safe for the web server reads.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
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

_issues: dict[str, dict] = {}
_websockets: set = set()
_main_loop: asyncio.AbstractEventLoop | None = None
_log_queue: asyncio.Queue | None = None


def init(loop: asyncio.AbstractEventLoop) -> None:
    global _main_loop, _log_queue
    _main_loop = loop
    _log_queue = asyncio.Queue()


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
    _push_event({"type": "log", "issue_id": issue_id, "message": message, "ts": _now()})


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
