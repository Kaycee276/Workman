# Workman

Workman is an autonomous AI agent that monitors GitHub for issues assigned to your account, solves them using Claude, and opens pull requests with the fixes — all without manual intervention.

A live web dashboard lets you watch every issue move through the pipeline in real time.

---

## What it does

1. **Polls GitHub** on a configurable interval for open issues assigned to your username
2. **Queues** all newly detected issues and displays them immediately in the dashboard
3. **Processes issues one at a time** through a 7-step pipeline:
   - `queued` — detected and waiting for its turn
   - `detected` — pipeline has started
   - `fetching` — pulls full issue details from GitHub
   - `forking` — forks the target repository into your account
   - `cloning` — clones the fork and creates a dedicated branch
   - `setup` — detects the project language and installs dependencies
   - `solving` — Claude analyzes the issue and writes the fix using tool use (read/write files, run commands, search code)
   - `pushing` — commits and pushes the branch, then opens a pull request
4. **Streams live updates** to the dashboard over WebSocket as each step completes

---

## Dashboard

The frontend is a React app that connects to the backend over WebSocket and shows:

- **Issues panel** — every tracked issue as a card with step-progress indicators, current status, and a link to the PR when done
- **Log console** — live structured log output, filterable per issue

Cards use a matte black / construction-yellow theme. Corner brackets on each card change colour by state: yellow (active), green (done), red (failed), gray (queued).

---

## Tech stack

| Layer | Stack |
|---|---|
| Backend | Python · FastAPI · Uvicorn · PyGithub |
| AI solver | Anthropic Claude (`claude-sonnet-4-6`) via tool use |
| Frontend | React · TypeScript · Vite |
| Transport | WebSocket (real-time pipeline events) |
| Deployment | Fly.io (`fly.toml` included) |

---

## Setup

### 1. Clone and configure

```bash
cd backend
cp .env.example .env
```

Edit `.env`:

```env
GITHUB_TOKEN=ghp_...        # needs repo + read:org scopes
GITHUB_USERNAME=your_handle
ANTHROPIC_API_KEY=sk-ant-...
POLL_INTERVAL=300            # seconds between GitHub polls (minimum 3)
WATCH_ORGS=org1,org2         # leave empty to watch all orgs
WORKDIR=/tmp/workman         # where repos are cloned
```

### 2. Run the backend

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

The API and WebSocket server starts on `http://localhost:8000`.

### 3. Run the frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

---

## How the solver works

When an issue reaches the `solving` step, Claude receives the issue title and description along with a set of tools it can use against the cloned repository:

| Tool | What it does |
|---|---|
| `read_file` | Read any file in the repo |
| `write_file` | Create or overwrite files |
| `list_files` | List directory contents |
| `search_code` | Grep for patterns across the codebase |
| `run_command` | Execute shell commands (tests, builds, linters) |
| `finish` | Signal completion and return a fix summary |

Claude iterates up to 40 times, exploring and editing until it calls `finish`. The fix summary is used as the PR description.

---

## CLI flags

```bash
python main.py          # Start server + polling loop (default)
python main.py --once   # Run one poll cycle and exit (no web server)
```

---

## Security

### What the `run_command` tool can do

When Claude solves an issue it has access to a `run_command` tool that executes arbitrary shell commands inside the cloned repository directory. This is intentional — Claude needs to run tests, builds, and linters — but it means **the process has the same filesystem access as the user running Workman**.

### Mitigations in place

| Mitigation | Detail |
|---|---|
| Credential stripping | `GITHUB_TOKEN`, `ANTHROPIC_API_KEY`, and other secrets are removed from the subprocess environment before any Claude-issued command runs, preventing them from being read back via `echo`/`env` |
| Command blocklist | Patterns for destructive commands are rejected before execution: `rm -rf` on system paths, `curl`/`wget` piped to a shell, writes to `/etc/` or `/root/`, `chmod` on system paths |
| Working directory | Every command runs with `cwd` set to the repo clone path |
| Timeout | Commands are killed after 180 seconds |

### What is NOT mitigated

- There is no filesystem namespace isolation. A creative command can still read world-readable host files or write to directories the process user owns.
- The blocklist is pattern-based and can potentially be bypassed by obfuscation.
- The web dashboard has no authentication — anyone who can reach the URL can see all issue activity.

### Recommendations

- **Do not run Workman as root.**
- **Do not run it on a machine that holds sensitive data** beyond the credentials already stripped from the environment.
- Treat it as a trusted personal tool, not a multi-tenant service.
- For stronger isolation, replace `NativeRunner` with a container-backed implementation and run on a dedicated VM or ephemeral sandbox.

---

## Deployment (Fly.io)

```bash
fly launch   # first time
fly deploy   # subsequent deploys
```

Set secrets on Fly instead of using a `.env` file:

```bash
fly secrets set GITHUB_TOKEN=... GITHUB_USERNAME=... ANTHROPIC_API_KEY=...
```
