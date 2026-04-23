import json
import logging
import shutil
from pathlib import Path

import config
from src import state
from src.drips.models import DripsIssue
from src.github.client import GitHubClient
from src.docker.manager import NativeRunner, clone_repo, detect_language, detect_projects, push_and_commit
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

    # Run every lint/typecheck/format-check script the project defines. CI
    # usually invokes a subset; running them all catches whatever CI catches
    # without us having to know the project's naming convention.
    lint_keywords = ("lint", "typecheck", "type-check", "format:check", "format-check", "check-format", "check:format")
    skip_keywords = ("fix", "write", "watch", "dev", "format:write")

    def _is_check_script(name: str) -> bool:
        n = name.lower()
        if any(s in n for s in skip_keywords):
            return False
        return any(k in n for k in lint_keywords) or n in ("check", "checks")

    typecheck_covered = False
    for name in sorted(scripts):
        if not _is_check_script(name):
            continue
        label = f"npm run {name}"
        checks.append((label, f"npm run {name} --silent 2>&1", 300))
        n = name.lower()
        if "typecheck" in n or "type-check" in n or n in ("check", "tsc"):
            typecheck_covered = True

    # Fallback: if no script covers typechecking but tsconfig is present, run tsc.
    if not typecheck_covered and (repo / "tsconfig.json").exists():
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


def _modified_projects(repo_root: Path, projects: list[tuple[str, Path]]) -> list[tuple[str, Path]]:
    """Return only the projects whose files were actually modified by the solver.

    Avoids running (slow, possibly unrelated) verification on a subproject the
    solver never touched — e.g. running `cargo test` on Soroban contracts when
    the fix only changed the Node server. Each changed file is attributed to
    its DEEPEST containing project so server/foo.js counts only for the Node
    project, not also for the rust project at the repo root.

    Falls back to ALL projects if git diff is unavailable.
    """
    runner = NativeRunner(repo_root)
    r = runner._run("git diff --name-only HEAD 2>&1", timeout=30)
    if r["exit_code"] != 0:
        logger.warning(f"git diff failed ({r['exit_code']}); running verification on all projects")
        return projects
    changed_rel = [line.strip() for line in r["stdout"].splitlines() if line.strip()]
    if not changed_rel:
        return []

    changed_abs = [(repo_root / p).resolve() for p in changed_rel]
    by_depth = sorted(
        projects, key=lambda lp: len(lp[1].resolve().parts), reverse=True
    )

    affected_keys: set[str] = set()
    for f in changed_abs:
        for _lang, subdir in by_depth:
            try:
                f.relative_to(subdir.resolve())
            except ValueError:
                continue
            affected_keys.add(str(subdir.resolve()))
            break

    return [
        (lang, subdir) for lang, subdir in projects
        if str(subdir.resolve()) in affected_keys
    ]


def _run_verification_multi(
    repo_root: Path, projects: list[tuple[str, Path]]
) -> str | None:
    """Run verification across every modified project. Labels each project in
    the failure report so the solver knows which subdir to fix."""
    targets = _modified_projects(repo_root, projects)
    if not targets:
        logger.info("No files modified by solver; verification gate skipped")
        return None

    failures: list[str] = []
    for lang, subdir in targets:
        sub_runner = NativeRunner(subdir)
        err = _run_verification(sub_runner, lang)
        if not err:
            continue
        try:
            rel = subdir.resolve().relative_to(repo_root.resolve())
            label = str(rel) if str(rel) != "." else "(root)"
        except ValueError:
            label = str(subdir)
        failures.append(f"### Project: {label} [{lang}] ###\n{err}")

    return "\n\n".join(failures) if failures else None


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

        # 4. Setup environment for every detected project (monorepo-aware).
        projects = detect_projects(repo_path)
        if not projects:
            projects = [(detect_language(repo_path), repo_path)]

        project_labels = ", ".join(
            f"{lang} @ {subdir.relative_to(repo_path) if subdir != repo_path else '.'}"
            for lang, subdir in projects
        )
        _step(iid, "setup", f"Detected projects: {project_labels}. Installing dependencies...")

        all_setup_warnings: list[dict] = []
        for lang, subdir in projects:
            sub_runner = NativeRunner(subdir)
            all_setup_warnings.extend(sub_runner.setup(lang))

        if all_setup_warnings:
            tagged = "; ".join(
                f"[{w.get('kind', 'UNKNOWN')}] {w.get('detail', 'no details')}"
                for w in all_setup_warnings
            )
            state.log(iid, f"Setup complete with warnings: {tagged}")
        else:
            state.log(iid, "Setup complete")

        # Preflight: every required language present must have its toolchain.
        for lang, _subdir in projects:
            required = _VERIFY_BINARIES.get(lang)
            if lang in _REQUIRED_VERIFY and required and not shutil.which(required):
                raise RuntimeError(
                    f"{required} not on PATH — refusing to ship an unverified {lang} fix"
                )

        available_tools = sorted(
            {name for name in _VERIFY_BINARIES.values() if shutil.which(name)}
        )

        # 5. Claude solver — rooted at repo_path; solver navigates into subdirs itself.
        _step(iid, "solving", "Claude is analyzing the issue and writing the fix...")
        runner = NativeRunner(repo_path)
        solver = IssueSolver(runner, repo_path)
        fix_summary = solver.solve(issue.title, issue.description, available_tools=available_tools)

        # 5.5. Verify the fix across every modified project.
        _step(iid, "verifying", "Running verification...")
        verification_error = _run_verification_multi(repo_path, projects)
        if verification_error:
            _step(iid, "re-solving", "Verification failed, continuing solver with errors (5 min budget)...")
            fix_summary = solver.continue_after_verification(verification_error)
            # After the retry pass, push whatever the solver built — trust the
            # solver's work and let CI evaluate. No second gate, no PR-body
            # disclosure.

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
