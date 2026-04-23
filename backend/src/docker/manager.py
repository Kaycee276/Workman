import logging
import os
import re
import shlex
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

# Explicit env vars never passed to Claude-executed subprocesses.
_STRIP_ENV_EXPLICIT: frozenset[str] = frozenset({
    "GITHUB_TOKEN",
    "ANTHROPIC_API_KEY",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_ACCESS_KEY_ID",
    "AWS_SESSION_TOKEN",
    "DATABASE_URL",
    "SECRET_KEY",
    "RENDER_API_KEY",
})

# Any var whose name ends with one of these suffixes is also stripped.
_SENSITIVE_SUFFIXES: tuple[str, ...] = (
    "_KEY", "_SECRET", "_TOKEN", "_PASSWORD", "_CREDENTIAL", "_API_KEY",
)

# Patterns that are blocked outright — destruction outside the repo or
# pipe-to-shell attacks that could exfiltrate secrets or modify the host.
_BLOCKED: list[re.Pattern] = [
    re.compile(r"rm\s+-\S*r\S*\s+[/~]"),                        # rm -rf / or ~/
    re.compile(r"rm\s+-\S*r\S*\s+\.\."),                        # rm -rf ../
    re.compile(r"(curl|wget)\s+.*\|\s*(\S+/)?(ba|z|da)?sh\b"),  # piped to any sh variant
    re.compile(r"(curl|wget)\s+.*\|\s*(\S+/)?python[23]?"),     # piped to python/python3
    re.compile(r"(curl|wget)\s+.*\|\s*(\S+/)?perl"),            # piped to perl
    re.compile(r"(curl|wget)\s+.*\|\s*(\S+/)?ruby"),            # piped to ruby
    re.compile(r"(curl|wget)\s+.*-o\s+/tmp/.*&&.*sh"),          # download-then-exec via /tmp
    re.compile(r">\s*/etc/"),                                    # overwrite system files
    re.compile(r">\s*/root/"),
    re.compile(r"chmod\s+.*\s+/"),                               # chmod on system paths
    re.compile(r"\bdd\s+.*\bof=/dev/"),                         # dd to block device
    re.compile(r"\bmkfs\b"),                                     # filesystem format
]



def _safe_env() -> dict[str, str]:
    """Return os.environ with credentials removed."""
    def _is_sensitive(name: str) -> bool:
        upper = name.upper()
        return upper in _STRIP_ENV_EXPLICIT or any(upper.endswith(s) for s in _SENSITIVE_SUFFIXES)
    return {k: v for k, v in os.environ.items() if not _is_sensitive(k)}


def _read_tail(fileobj, max_bytes: int) -> str:
    fileobj.seek(0, os.SEEK_END)
    size = fileobj.tell()
    if size <= max_bytes:
        fileobj.seek(0)
        return fileobj.read().decode("utf-8", errors="replace")
    fileobj.seek(size - max_bytes)
    tail = fileobj.read().decode("utf-8", errors="replace")
    return f"[truncated, {size - max_bytes} bytes dropped]\n{tail}"


def _check_command(command: str) -> None:
    """Raise if command matches a blocked pattern."""
    for pattern in _BLOCKED:
        if pattern.search(command):
            raise PermissionError(f"Blocked command pattern '{pattern.pattern}': {command}")

LANG_SETUP_CMDS: dict[str, list[str]] = {
    "python": [
        "python3 -m pip install --upgrade pip -q 2>/dev/null || true",
        "[ -f requirements.txt ] && python3 -m pip install -r requirements.txt -q || true",
        "[ -f pyproject.toml ] && python3 -m pip install -e . -q 2>/dev/null || true",
        "[ -f setup.py ] && python3 -m pip install -e . -q 2>/dev/null || true",
        "[ -f pyproject.toml ] && python3 -m pip install -e '.[dev]' -q 2>/dev/null || true",
    ],
    "go": [
        "[ -f go.mod ] && go mod download 2>/dev/null || true",
    ],
    "node": [
        # Try the project's preferred package manager, fall back to npm if the
        # preferred binary is absent. Ensures node_modules exists even when the
        # host doesn't have pnpm/yarn/bun installed.
        (
            "if [ -f pnpm-lock.yaml ] && command -v pnpm >/dev/null; then "
            "pnpm install --frozen-lockfile; "
            "elif [ -f yarn.lock ] && command -v yarn >/dev/null; then "
            "yarn install --frozen-lockfile; "
            "elif [ -f bun.lockb ] && command -v bun >/dev/null; then "
            "bun install; "
            "elif [ -f package-lock.json ]; then "
            "npm ci; "
            "elif [ -f package.json ]; then "
            "npm install; "
            "fi"
        ),
    ],
    "rust": [
        "[ -f Cargo.toml ] && cargo fetch 2>/dev/null || true",
    ],
    "default": [],
}

SETUP_TIMEOUT = 600

# Cap per-stream captured output. 2GB host absorbs half-MB comfortably; this is
# big enough for full test output but still bounds runaway progress spam.
MAX_CAPTURE_BYTES = 512 * 1024


def detect_language(repo_path: Path) -> str:
    indicators = {
        "python": ["requirements.txt", "pyproject.toml", "setup.py", "setup.cfg"],
        "node": ["package.json"],
        "rust": ["Cargo.toml"],
        "go": ["go.mod"],
        "java": ["pom.xml", "build.gradle"],
        "ruby": ["Gemfile"],
        "php": ["composer.json"],
    }
    for lang, files in indicators.items():
        if any((repo_path / f).exists() for f in files):
            return lang
    return "default"


class NativeRunner:
    """
    Runs commands directly on the host — no Docker required.

    SECURITY MODEL
    --------------
    Commands are executed via ``bash -c`` inside the cloned repository
    directory with the following mitigations applied:

    - Credentials stripped: GITHUB_TOKEN, ANTHROPIC_API_KEY and other
      secrets are removed from the subprocess environment via _safe_env(),
      so they cannot be read back through echo/env/printenv.
    - Blocklist: _check_command() rejects patterns known to be destructive
      (rm -rf on system paths, curl/wget piped to shell, writes to /etc or
      /root, chmod on system paths) before execution.
    - Working directory: cwd is always set to the repo clone path, so
      relative paths stay inside the repo.
    - Timeout: each command is killed after 600 seconds.

    What is NOT mitigated
    ---------------------
    - There is no filesystem namespace isolation (no chroot / container).
      A sufficiently creative command can still read host files that are
      world-readable or write to directories the process user owns.
    - The blocklist is pattern-based and can be bypassed by obfuscation.
    - Do not run Workman as root or on a machine that holds sensitive data
      beyond what is already stripped from the environment.
    - For stronger isolation, replace NativeRunner with a container-backed
      implementation and run on a dedicated VM or ephemeral sandbox.
    """

    def __init__(self, repo_path: Path):
        self.repo_path = repo_path

    def setup(self, language: str) -> list[dict]:
        """Run setup commands. Returns structured warnings for any failures.

        Each warning is {"kind": str, "cmd": str, "detail": str} where kind is
        one of: "timeout", "network", "exit". Use kind to aggregate patterns
        across runs (e.g. repeated "network" across repos means a flaky host).
        """
        warnings: list[dict] = []
        for cmd in LANG_SETUP_CMDS.get(language, []):
            logger.debug(f"Setup: {cmd}")
            r = self._run(cmd, timeout=SETUP_TIMEOUT)
            if r["exit_code"] == 0:
                continue
            stderr = r["stderr"][:300]
            if "timed out" in stderr.lower():
                kind = "timeout"
            elif any(s in stderr.lower() for s in ("network is unreachable", "dns", "name resolution", "could not resolve", "connection refused", "connection timed out")):
                kind = "network"
            else:
                kind = "exit"
            warning = {"kind": kind, "cmd": cmd, "detail": stderr}
            logger.warning(f"Setup step failed [{kind}] ({r['exit_code']}): {stderr}")
            warnings.append(warning)
        return warnings

    def exec(self, command: str) -> dict:
        try:
            _check_command(command)
        except PermissionError as e:
            logger.warning(str(e))
            return {"exit_code": 1, "stdout": "", "stderr": str(e)}
        return self._run(command)

    def _run(self, command: str, timeout: int = 600) -> dict:
        try:
            with tempfile.TemporaryFile() as out_f, tempfile.TemporaryFile() as err_f:
                result = subprocess.run(
                    ["bash", "-c", command],
                    stdout=out_f,
                    stderr=err_f,
                    cwd=str(self.repo_path),
                    env=_safe_env(),
                    timeout=timeout,
                )
                return {
                    "exit_code": result.returncode,
                    "stdout": _read_tail(out_f, MAX_CAPTURE_BYTES),
                    "stderr": _read_tail(err_f, MAX_CAPTURE_BYTES),
                }
        except subprocess.TimeoutExpired:
            return {"exit_code": 1, "stdout": "", "stderr": f"Command timed out after {timeout}s"}
        except Exception as e:
            return {"exit_code": 1, "stdout": "", "stderr": str(e)}


def _write_askpass(token: str) -> str:
    """Write a temp credential-helper script. Returns path; caller must delete."""
    script = (
        "#!/bin/sh\n"
        'case "$1" in\n'
        "  *Username*) echo x-access-token ;;\n"
        f"  *) echo {shlex.quote(token)} ;;\n"
        "esac\n"
    )
    fd, path = tempfile.mkstemp(suffix=".sh", prefix="wm_cred_")
    try:
        os.write(fd, script.encode())
    finally:
        os.close(fd)
    os.chmod(path, stat.S_IRWXU)  # 700 — only owner can read/execute
    return path


def clone_repo(clone_url: str, dest: Path, branch: str | None = None, token: str | None = None) -> None:
    import config as _cfg

    # Guard against path traversal — dest must live inside WORKDIR
    workdir = Path(_cfg.WORKDIR).resolve()
    dest_resolved = dest.resolve()
    if not str(dest_resolved).startswith(str(workdir)):
        raise ValueError(f"Clone destination {dest} is outside WORKDIR {workdir}")

    # Strip any token already embedded in the URL
    clean_url = re.sub(r"https://[^@]+@github\.com", "https://github.com", clone_url)

    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    env = _safe_env()
    askpass_path = None
    if token:
        askpass_path = _write_askpass(token)
        env["GIT_ASKPASS"] = askpass_path
        env["GIT_TERMINAL_PROMPT"] = "0"

    try:
        result = subprocess.run(
            ["git", "clone", "--depth=1", clean_url, str(dest)],
            capture_output=True, text=True, env=env,
        )
    finally:
        if askpass_path:
            try:
                os.unlink(askpass_path)
            except OSError:
                pass

    if result.returncode != 0:
        raise RuntimeError(f"git clone failed: {result.stderr}")

    # Verify the clone actually produced a repo
    if not (dest / ".git").exists():
        raise RuntimeError("git clone succeeded but .git directory is missing")

    # Embed credentials into the local .git/config for subsequent pushes.
    # File I/O keeps the token out of process args / ps aux.
    if token:
        authed_url = clean_url.replace("https://github.com/", f"https://x-access-token:{token}@github.com/")
        git_cfg = dest / ".git" / "config"
        text = git_cfg.read_text()
        text = text.replace(f"url = {clean_url}", f"url = {authed_url}")
        git_cfg.write_text(text)
        os.chmod(str(git_cfg), 0o600)  # owner read/write only

    if branch:
        result = subprocess.run(
            ["git", "checkout", "-b", branch],
            capture_output=True, text=True, cwd=str(dest),
        )
        if result.returncode != 0:
            raise RuntimeError(f"git checkout -b failed: {result.stderr}")

    logger.info("Clone complete")


def push_and_commit(repo_path: Path, branch: str, commit_message: str) -> None:
    cmds = [
        ["git", "config", "user.email", "workman@bot.local"],
        ["git", "config", "user.name", "Workman Bot"],
        ["git", "add", "-A"],
        ["git", "commit", "-m", commit_message],
        ["git", "push", "origin", branch],
    ]
    for cmd in cmds:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(repo_path))
        if result.returncode != 0:
            if "nothing to commit" in result.stdout + result.stderr:
                logger.info("Nothing new to commit")
                continue
            raise RuntimeError(f"{' '.join(cmd)} failed: {result.stderr}")
    logger.info(f"Pushed branch {branch}")
