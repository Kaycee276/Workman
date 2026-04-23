import json
import logging
import shutil
from pathlib import Path

import config
from src import state
from src.drips.models import DripsIssue
from src.github.client import GitHubClient
from src.docker.manager import NativeRunner, clone_repo, detect_language, push_and_commit
from src.solver.agent import IssueSolver

logger = logging.getLogger(__name__)

# Languages whose fixes can't be sanity-checked without a compiler/type-checker.
# If the binary is absent we abort rather than ship an unverifiable PR.
_VERIFY_BINARIES: dict[str, str] = {
    "rust": "cargo",
    "node": "npm",
    "go": "go",
    "python": "python3",
}
_REQUIRED_VERIFY: frozenset[str] = frozenset({"rust", "node"})


def _node_checks(repo: Path) -> list[tuple[str, str, int]]:
    pkg = repo / "package.json"
    scripts: dict = {}
    if pkg.exists():
        try:
            scripts = (json.loads(pkg.read_text()).get("scripts") or {})
        except Exception:
            scripts = {}

    checks: list[tuple[str, str, int]] = []

    if "test" in scripts:
        checks.append(("test", "npm test --silent 2>&1", 900))

    if "lint" in scripts:
        checks.append(("lint", "npm run lint --silent 2>&1", 300))

    if "typecheck" in scripts:
        checks.append(("typecheck", "npm run typecheck --silent 2>&1", 300))
    elif "type-check" in scripts:
        checks.append(("typecheck", "npm run type-check --silent 2>&1", 300))
    elif (repo / "tsconfig.json").exists():
        checks.append(("typecheck", "npx --no-install tsc --noEmit 2>&1", 300))

    return checks


def _rust_checks() -> list[tuple[str, str, int]]:
    return [
        ("test", "cargo test --no-fail-fast 2>&1", 1500),
        ("clippy", "cargo clippy --all-targets --all-features -- -D warnings 2>&1", 600),
        ("fmt", "cargo fmt --all -- --check 2>&1", 60),
    ]


def _python_checks(repo: Path) -> list[tuple[str, str, int]]:
    checks: list[tuple[str, str, int]] = []
    has_tests = (
        (repo / "tests").is_dir()
        or any(repo.glob("test_*.py"))
        or any(repo.glob("*_test.py"))
    )
    if has_tests:
        checks.append(("pytest", "python3 -m pytest -x --no-header 2>&1", 600))

    if (repo / "pyproject.toml").exists() or (repo / "ruff.toml").exists() or (repo / ".ruff.toml").exists():
        checks.append(("ruff", "ruff check . 2>&1", 60))

    if (repo / "mypy.ini").exists() or _has_mypy_config(repo):
        checks.append(("mypy", "mypy . 2>&1", 300))

    return checks


def _has_mypy_config(repo: Path) -> bool:
    pyproject = repo / "pyproject.toml"
    if not pyproject.exists():
        return False
    try:
        return "[tool.mypy]" in pyproject.read_text()
    except Exception:
        return False


def _go_checks() -> list[tuple[str, str, int]]:
    return [
        ("test", "go test ./... 2>&1", 600),
        ("vet", "go vet ./... 2>&1", 120),
        ("build", "go build ./... 2>&1", 300),
    ]


def _verification_commands(repo: Path, language: str) -> list[tuple[str, str, int]]:
    if language == "node":
        return _node_checks(repo)
    if language == "rust":
        return _rust_checks()
    if language == "python":
        return _python_checks(repo)
    if language == "go":
        return _go_checks()
    return []


def _run_verification(runner: NativeRunner, language: str) -> str | None:
    """Run tests, lint, and typecheck. Return aggregated failure report, or None.

    Runs each applicable check sequentially. Collects failures instead of
    short-circuiting so the solver sees the full picture on its retry pass.
    """
    checks = _verification_commands(runner.repo_path, language)
    if not checks:
        logger.info(f"No verification configured for language={language}; skipping gate")
        return None

    failures: list[str] = []
    for label, cmd, timeout in checks:
        logger.info(f"Verification [{label}]: {cmd}")
        r = runner._run(cmd, timeout=timeout)
        if r["exit_code"] != 0:
            out = r["stdout"].strip()
            err = r["stderr"].strip()
            body = "\n".join(p for p in (out, err) if p)
            failures.append(f"=== {label} failed (exit {r['exit_code']}) ===\n{body}")
        else:
            logger.info(f"Verification [{label}]: passed")

    if failures:
        return "\n\n".join(failures)
    return None


def _step(issue_id: str, step: str, msg: str) -> None:
    logger.info(msg)
    state.upsert_issue(issue_id, step=step)
    state.log(issue_id, msg)


def _cleanup_path(path: Path) -> None:
    if not path.exists():
        return
    try:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)
    except Exception as e:
        logger.warning(f"Failed to clean {path}: {e}")


def run_pipeline(issue: DripsIssue) -> str:
    iid = issue.id
    state.upsert_issue(iid, step="detected")
    state.log(iid, f"Pipeline started for {iid}")

    gh = GitHubClient()
    repo_path = Path(config.WORKDIR) / f"{issue.repo_owner}_{issue.repo_name}_{issue.issue_number}"
    _cleanup_path(repo_path)

    try:
        # 1. Fetch GitHub issue details
        _step(iid, "fetching", "Fetching issue details from GitHub...")
        details = gh.get_issue_details(issue.repo_owner, issue.repo_name, issue.issue_number)
        issue.title = details["title"]
        issue.description = details["body"]
        issue.labels = details["labels"]
        state.upsert_issue(iid, title=issue.title)
        state.log(iid, f"Issue: {issue.title}")

        # 2. Fork
        _step(iid, "forking", f"Forking {issue.repo_owner}/{issue.repo_name}...")
        forked_repo = gh.fork_repo(issue.repo_owner, issue.repo_name)
        source_repo = gh.g.get_repo(f"{issue.repo_owner}/{issue.repo_name}")
        state.log(iid, f"Fork ready: {forked_repo.full_name}")

        # Fast-forward the fork to upstream HEAD — forks don't auto-sync, so
        # a repo we forked weeks ago for a past issue would otherwise send
        # Claude working against stale code.
        if gh.sync_fork(forked_repo):
            state.log(iid, "Fork synced with upstream")
        else:
            state.log(iid, "Fork sync skipped (diverged or API error) — continuing with existing fork state")

        # 3. Clone
        branch_name = gh.make_branch_name(issue.issue_number, issue.title)

        # Delete any stale branch from a previous attempt on this issue. If
        # a human closed an earlier PR to signal "retry", its branch still
        # lives on the fork and would cause a non-fast-forward push later.
        try:
            forked_repo.get_git_ref(f"heads/{branch_name}").delete()
            state.log(iid, f"Deleted stale branch {branch_name} from previous attempt")
        except Exception:
            pass

        _step(iid, "cloning", f"Cloning fork (branch: {branch_name})...")
        clone_repo(gh.get_clone_url(forked_repo), repo_path, branch=branch_name, token=config.GITHUB_TOKEN)
        state.log(iid, "Clone complete")

        # 4. Setup environment natively
        language = detect_language(repo_path)
        _step(iid, "setup", f"Detected language: {language}. Installing dependencies...")
        runner = NativeRunner(repo_path)
        setup_warnings = runner.setup(language)
        if setup_warnings:
            tagged = "; ".join(
                f"[{w.get('kind', 'UNKNOWN')}] {w.get('detail', 'no details')}"
                for w in setup_warnings
            )
            state.log(iid, f"Setup complete with warnings: {tagged}")
        else:
            state.log(iid, "Setup complete")

        # Preflight: abort if we can't even sanity-check a fix for this language.
        required = _VERIFY_BINARIES.get(language)
        if language in _REQUIRED_VERIFY and required and not shutil.which(required):
            raise RuntimeError(
                f"{required} not on PATH — refusing to ship an unverified {language} fix"
            )

        available_tools = sorted(
            name for name in _VERIFY_BINARIES.values() if shutil.which(name)
        )

        # 5. Claude solver
        _step(iid, "solving", "Claude is analyzing the issue and writing the fix...")
        solver = IssueSolver(runner, repo_path)
        fix_summary = solver.solve(issue.title, issue.description, available_tools=available_tools)

        # 5.5. Verify the fix
        _step(iid, "verifying", "Running verification...")
        verification_error = _run_verification(runner, language)
        if verification_error:
            _step(iid, "re-solving", "Verification failed, re-analyzing with errors...")
            issue.description += f"\n\nVerification failed. Please fix the following errors and ensure tests pass:\n{verification_error}"
            fix_summary = solver.solve(issue.title, issue.description, available_tools=available_tools)
            # Verify again
            verification_error = _run_verification(runner, language)
            if verification_error:
                raise RuntimeError(f"Verification failed after retry: {verification_error}")

        state.log(iid, f"Fix complete: {fix_summary}")

        # 6. Push
        _step(iid, "pushing", "Committing and pushing branch...")
        push_and_commit(repo_path, branch_name, f"fix: resolve issue #{issue.issue_number} - {issue.title}")
        state.log(iid, "Branch pushed")

        # 7. PR
        state.log(iid, "Creating pull request...")
        try:
            pr_url = gh.create_pull_request(
                source_repo=source_repo,
                fork_repo=forked_repo,
                branch=branch_name,
                issue_number=issue.issue_number,
                issue_title=issue.title,
                fix_summary=fix_summary,
            )
        except Exception as pr_exc:
            # Clean up the orphaned branch so it doesn't accumulate on the fork
            try:
                forked_repo.get_git_ref(f"heads/{branch_name}").delete()
                state.log(iid, f"Rolled back branch {branch_name} after PR failure")
            except Exception:
                pass
            raise pr_exc

        state.upsert_issue(iid, step="done", pr_url=pr_url)
        state.log(iid, f"PR created: {pr_url}")
        return pr_url

    except Exception as exc:
        state.upsert_issue(iid, failed=True, error=str(exc))
        state.log(iid, f"ERROR: {exc}")
        raise
    finally:
        _cleanup_path(repo_path)
