import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

LANG_SETUP_CMDS: dict[str, list[str]] = {
    "python": [
        "pip install --upgrade pip -q 2>/dev/null || true",
        "[ -f requirements.txt ] && pip install -r requirements.txt -q || true",
        "[ -f pyproject.toml ] && pip install -e . -q 2>/dev/null || true",
        "[ -f setup.py ] && pip install -e . -q 2>/dev/null || true",
    ],
    "node": [
        "[ -f package.json ] && npm install --silent 2>/dev/null || true",
    ],
    "rust": [],
    "go": [
        "[ -f go.mod ] && go mod download 2>/dev/null || true",
    ],
    "default": [],
}


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
    """Runs commands directly on the host — no Docker required."""

    def __init__(self, repo_path: Path):
        self.repo_path = repo_path

    def setup(self, language: str) -> None:
        for cmd in LANG_SETUP_CMDS.get(language, []):
            logger.debug(f"Setup: {cmd}")
            r = self._run(cmd)
            if r["exit_code"] != 0:
                logger.warning(f"Setup step non-zero ({r['exit_code']}): {r['stderr'][:200]}")

    def exec(self, command: str) -> dict:
        return self._run(command)

    def cleanup(self) -> None:
        pass

    def cleanup_all(self) -> None:
        pass

    def _run(self, command: str) -> dict:
        try:
            result = subprocess.run(
                ["bash", "-c", command],
                capture_output=True,
                text=True,
                cwd=str(self.repo_path),
                timeout=180,
            )
            return {
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        except subprocess.TimeoutExpired:
            return {"exit_code": 1, "stdout": "", "stderr": "Command timed out after 180s"}
        except Exception as e:
            return {"exit_code": 1, "stdout": "", "stderr": str(e)}


def clone_repo(clone_url: str, dest: Path, branch: str | None = None) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        ["git", "clone", "--depth=1", clone_url, str(dest)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git clone failed: {result.stderr}")

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
