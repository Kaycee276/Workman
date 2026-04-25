import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Required — validated at startup in main.py
GITHUB_TOKEN      = os.getenv("GITHUB_TOKEN")
GITHUB_USERNAME   = os.getenv("GITHUB_USERNAME")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GROQ_API_KEY      = os.getenv("GROQ_API_KEY")

POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "300"))

# Default to backend/work/ — persistent, private, not /tmp
WORKDIR = os.getenv("WORKDIR", str(Path(__file__).parent / "work"))

# Comma-separated GitHub org names to watch for assigned issues.
# e.g. "stellar,drips-network" — leave empty to watch all orgs (use with care).
WATCH_ORGS = os.getenv("WATCH_ORGS", "")

# Comma-separated GitHub org names to IGNORE, even when assigned. Useful for
# orgs whose issues Workman should never touch (day-job, private clients, etc).
EXCLUDE_ORGS = os.getenv("EXCLUDE_ORGS", "")

# Label that must be present on an issue for it to be picked up.
# e.g. "Stellar Wave" for Drips bounties. Leave empty to skip label filtering.
WATCH_LABEL = os.getenv("WATCH_LABEL", "")

# Optional dashboard auth token. If set, WebSocket and /api/status require
# ?token=<value>. Leave empty to disable auth (local / trusted environments only).
DASHBOARD_TOKEN = os.getenv("DASHBOARD_TOKEN", "")

# Comma-separated allowed CORS origins.
# Defaults to localhost only — set explicitly for any deployed frontend.
_raw_origins = os.getenv("ALLOWED_ORIGINS", "")
ALLOWED_ORIGINS = (
    [o.strip() for o in _raw_origins.split(",") if o.strip()]
    if _raw_origins.strip()
    else ["http://localhost:5173", "http://localhost:3000"]
)

MAX_SOLVER_ITERATIONS = 90
