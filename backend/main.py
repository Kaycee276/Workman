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
from src.github.client import GitHubClient
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
MAX_RETRIES = 3


def load_processed() -> tuple[set[str], dict[str, int]]:
    if STATE_FILE.exists():
        try:
            data = json.loads(STATE_FILE.read_text())
            if not isinstance(data, dict):
                raise ValueError("State file is not a JSON object")
            return set(data.get("processed", [])), data.get("failures", {})
        except (json.JSONDecodeError, ValueError):
            logger.warning(f"Corrupted {STATE_FILE}, starting with empty state")
    return set(), {}


def save_processed(processed: set[str], failures: dict[str, int]) -> None:
    STATE_FILE.write_text(json.dumps({"processed": list(processed), "failures": failures}, indent=2))


async def poll_loop(watcher: DripsWatcher, processed: set[str], failures: dict[str, int]) -> None:
    while True:
        await check_and_process(watcher, processed, failures)
        logger.info(f"Sleeping {config.POLL_INTERVAL}s until next check...")
        await asyncio.sleep(config.POLL_INTERVAL)


async def check_and_process(watcher: DripsWatcher, processed: set[str], failures: dict[str, int]) -> None:
    logger.info("Checking Drips for assigned issues...")
    state.log(None, "Polling Drips for assigned issues...")

    try:
        issues = await watcher.get_assigned_issues()
    except Exception as e:
        msg = f"Failed to fetch issues from Drips: {e}"
        logger.error(msg)
        state.log(None, f"ERROR: {msg}")
        return

    # Filter out already-processed and exhausted-retry issues
    candidates = [
        i for i in issues
        if i.id not in processed and failures.get(i.id, 0) < MAX_RETRIES
    ]
    for i in issues:
        if i.id not in processed and failures.get(i.id, 0) >= MAX_RETRIES:
            logger.warning(f"Skipping {i.id} — failed {failures[i.id]} times, giving up")

    if not candidates:
        logger.info("No new assigned issues.")
        state.log(None, "No new assigned issues found.")
        return

    # Cross-check against GitHub: skip any issue that already has a PR from us.
    # This survives server restarts even when state.json is missing (e.g. after Render redeploy).
    gh = GitHubClient()
    actionable = []
    for issue in candidates:
        try:
            pr_url = await asyncio.to_thread(
                gh.find_existing_pr, issue.repo_owner, issue.repo_name, issue.issue_number
            )
        except Exception as e:
            logger.warning(f"Failed to check existing PR for {issue.id}, will process anyway: {e}")
            pr_url = None
        if pr_url:
            logger.info(f"Skipping {issue.id} — PR already exists: {pr_url}")
            state.log(issue.id, f"PR already exists: {pr_url}")
            processed.add(issue.id)
            save_processed(processed, failures)
        else:
            actionable.append(issue)

    if not actionable:
        logger.info("All issues already have PRs.")
        state.log(None, "All candidates already have PRs — nothing to process.")
        return

    logger.info(f"New issues: {[i.id for i in actionable]}")

    # Register all new issues immediately so they appear in the dashboard as queued
    for issue in actionable:
        state.upsert_issue(issue.id, title=issue.title, step="queued")
        state.log(issue.id, f"Issue queued: {issue.title}")

    # Process one at a time
    for issue in actionable:
        logger.info(f"Processing {issue.id}...")
        try:
            pr_url = await asyncio.to_thread(run_pipeline, issue)
            logger.info(f"SUCCESS — PR: {pr_url}")
            processed.add(issue.id)
            failures.pop(issue.id, None)
            save_processed(processed, failures)
        except Exception as e:
            failures[issue.id] = failures.get(issue.id, 0) + 1
            save_processed(processed, failures)
            logger.error(f"Pipeline failed for {issue.id} (attempt {failures[issue.id]}/{MAX_RETRIES}): {e}", exc_info=True)
            state.log(issue.id, f"Attempt {failures[issue.id]}/{MAX_RETRIES} failed: {e}")


async def main_async(once: bool) -> None:
    _validate_config()

    loop = asyncio.get_event_loop()
    state.init(loop)

    # Route all Python log records into the WebSocket broadcaster too
    ws_handler = state.StateLogHandler()
    ws_handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))
    logging.getLogger().addHandler(ws_handler)

    watcher = DripsWatcher()
    processed, failures = load_processed()

    if once:
        await check_and_process(watcher, processed, failures)
        return

    # Run: broadcaster + web server + poll loop, all concurrently
    # ws_ping_interval=None disables the websockets library's built-in PING frames.
    # Render's proxy silently drops them, causing keepalive timeout errors on the server.
    # The frontend sends application-level "ping" text messages every 30s instead.
    config_uv = uvicorn.Config(
        app, host="0.0.0.0", port=8000, log_level="warning",
        ws_ping_interval=None,
    )
    server = uvicorn.Server(config_uv)

    await asyncio.gather(
        state.broadcaster(),
        server.serve(),
        poll_loop(watcher, processed, failures),
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
