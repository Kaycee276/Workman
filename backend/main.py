"""
Workman — autonomous Drips Wave issue solver with live dashboard.

Usage:
    python main.py          # Start web server + polling loop
    python main.py --once   # One poll cycle then exit (no web server)
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

import uvicorn

import config
from src import state
from src.drips.watcher import DripsWatcher
from src.pipeline import run_pipeline
from src.web.server import app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("workman.log"),
    ],
)
logger = logging.getLogger("workman")

STATE_FILE = Path("state.json")


def load_processed() -> set[str]:
    if STATE_FILE.exists():
        data = json.loads(STATE_FILE.read_text())
        return set(data.get("processed", []))
    return set()


def save_processed(processed: set[str]) -> None:
    STATE_FILE.write_text(json.dumps({"processed": list(processed)}, indent=2))


async def poll_loop(watcher: DripsWatcher, processed: set[str]) -> None:
    while True:
        await check_and_process(watcher, processed)
        logger.info(f"Sleeping {config.POLL_INTERVAL}s until next check...")
        await asyncio.sleep(config.POLL_INTERVAL)


async def check_and_process(watcher: DripsWatcher, processed: set[str]) -> None:
    logger.info("Checking Drips for assigned issues...")
    state.log(None, "Polling Drips for assigned issues...")

    try:
        issues = await watcher.get_assigned_issues()
    except Exception as e:
        msg = f"Failed to fetch issues from Drips: {e}"
        logger.error(msg)
        state.log(None, f"ERROR: {msg}")
        return

    new_issues = [i for i in issues if i.id not in processed]
    if not new_issues:
        logger.info("No new assigned issues.")
        state.log(None, "No new assigned issues found.")
        return

    logger.info(f"New issues: {[i.id for i in new_issues]}")

    # Register all new issues immediately so they appear in the dashboard as queued
    for issue in new_issues:
        state.upsert_issue(issue.id, title=issue.title, step="queued")
        state.log(issue.id, f"Issue queued: {issue.title}")

    # Process one at a time
    for issue in new_issues:
        logger.info(f"Processing {issue.id}...")
        try:
            pr_url = await asyncio.to_thread(run_pipeline, issue)
            logger.info(f"SUCCESS — PR: {pr_url}")
            processed.add(issue.id)
            save_processed(processed)
        except Exception as e:
            logger.error(f"Pipeline failed for {issue.id}: {e}", exc_info=True)


async def main_async(once: bool) -> None:
    _validate_config()

    loop = asyncio.get_event_loop()
    state.init(loop)

    # Route all Python log records into the WebSocket broadcaster too
    ws_handler = state.StateLogHandler()
    ws_handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))
    logging.getLogger().addHandler(ws_handler)

    watcher = DripsWatcher()
    processed = load_processed()

    if once:
        await check_and_process(watcher, processed)
        return

    # Run: broadcaster + web server + poll loop, all concurrently
    config_uv = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="warning")
    server = uvicorn.Server(config_uv)

    await asyncio.gather(
        state.broadcaster(),
        server.serve(),
        poll_loop(watcher, processed),
    )


def _validate_config() -> None:
    missing = [k for k, v in [
        ("GITHUB_TOKEN", config.GITHUB_TOKEN),
        ("GITHUB_USERNAME", config.GITHUB_USERNAME),
        ("ANTHROPIC_API_KEY", config.ANTHROPIC_API_KEY),
    ] if not v]
    if missing:
        logger.error(f"Missing env vars: {', '.join(missing)} — copy .env.example to .env")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="One poll cycle, no web server")
    args = parser.parse_args()
    asyncio.run(main_async(once=args.once))
