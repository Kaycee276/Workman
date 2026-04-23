import json
import logging
import shlex
from pathlib import Path

import anthropic

import config
from src.docker.manager import NativeRunner

logger = logging.getLogger(__name__)

TOOLS = [
    {
        "name": "read_file",
        "description": "Read the contents of a file in the repository.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path relative to repo root"}
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write (create or overwrite) a file in the repository.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path relative to repo root"},
                "content": {"type": "string", "description": "Full file content"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "list_files",
        "description": "List files and directories at a given path.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path relative to repo root. Use '.' for root."}
            },
            "required": ["path"],
        },
    },
    {
        "name": "search_code",
        "description": "Search for a text pattern across files (grep). Returns matching lines.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Text or regex to search for"},
                "path": {"type": "string", "description": "Directory or file to search in (default: '.')", "default": "."},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "run_command",
        "description": "Run a shell command in the repository directory (e.g. run tests, install packages, build).",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to execute"}
            },
            "required": ["command"],
        },
    },
    {
        "name": "finish",
        "description": "Call this when the fix is complete and tests pass. Provide a summary of changes.",
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "description": "Human-readable summary of what was changed and why"}
            },
            "required": ["summary"],
        },
    },
]

SYSTEM_PROMPT = """\
You are Workman, an autonomous software engineer. You have been assigned a GitHub issue to fix.

Your workflow:
1. Read the issue description carefully
2. Explore the repository structure to understand the codebase
3. Find the relevant code that needs to change
4. Make the necessary edits
5. Verify your fix by running BOTH tests AND the project's lint/typecheck checks. Running only one is not enough — most PRs fail CI on lint or typecheck, not tests. Read `package.json` scripts to find the right command names (e.g. `npm run lint`, `npm run typecheck`, `npm run format:check`). For Python: `python3 -m pytest`, `ruff check .`, `mypy .`. For Go: `go test ./...`, `go vet ./...`. For Rust: `cargo test`, `cargo clippy -- -D warnings`, `cargo fmt -- --check`.
6. If any check fails, debug and iterate until ALL checks pass
7. Only when every applicable check passes, call `finish`

Rules:
- Only change what is necessary to fix the issue — do not refactor unrelated code
- Run tests AND lint AND typecheck before finishing. Skipping any of these is the #1 reason CI rejects the resulting PR.
- Never use the `any` type in TypeScript; always use proper types instead.
- Write clean, idiomatic code matching the project's style
- Do not add comments unless the logic is genuinely non-obvious
- Be decisive: once all checks pass, call `finish`. Do not loop retrying the same failing command more than twice.
- The `finish` summary describes the code change only. Do not mention verification results.

Finishing — read carefully:
- Run verification (tests, type-check, build, or lint) and ensure it passes before calling `finish`.
- If verification fails, fix the code and re-run verification.
- Do NOT call `finish` until verification passes.
- Do NOT after that point: add documentation, write extra test cases, chase coverage numbers, refactor the fix, or tweak unrelated files.
- After you call `finish`, the pipeline runs lint, typecheck, and tests again as an independent gate. If that gate fails, you will be invoked a second time with the failures — so it is in your interest to run those same checks yourself first.

Environment:
- Dependencies are pre-installed before you start. `node_modules`, Python packages, cargo registry etc. are ready. You may install additional dev-only packages if a verification tool is missing (e.g. `npm install --no-save <tool>`), but do not touch the committed lockfile.
"""


PRIMARY_MODEL = "claude-opus-4-7"
FALLBACK_MODEL = "claude-sonnet-4-6"

# Keep the last N tool-result turns verbatim; older ones are shrunk to a stub.
# Stale tool output doesn't help the model reason and balloons memory / tokens.
TOOL_RESULT_WINDOW = 25
TOOL_RESULT_STUB_LEN = 120


def _trim_old_tool_results(messages: list[dict]) -> None:
    tool_turns = [
        i for i, m in enumerate(messages)
        if m["role"] == "user"
        and isinstance(m["content"], list)
        and any(isinstance(b, dict) and b.get("type") == "tool_result" for b in m["content"])
    ]
    if len(tool_turns) <= TOOL_RESULT_WINDOW:
        return
    for idx in tool_turns[:-TOOL_RESULT_WINDOW]:
        for block in messages[idx]["content"]:
            if not (isinstance(block, dict) and block.get("type") == "tool_result"):
                continue
            content = block.get("content", "")
            if isinstance(content, str) and len(content) > TOOL_RESULT_STUB_LEN:
                first_line = content.split("\n", 1)[0][:TOOL_RESULT_STUB_LEN]
                block["content"] = f"[truncated, {len(content)} chars] {first_line}"


class IssueSolver:
    def __init__(self, runner: NativeRunner, repo_path: Path):
        self.client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        self.runner = runner
        self.repo_path = repo_path
        self.model = PRIMARY_MODEL

    def _create(self, messages: list[dict]):
        try:
            return self.client.messages.create(
                model=self.model,
                max_tokens=8096,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages,
            )
        except (anthropic.NotFoundError, anthropic.BadRequestError) as e:
            if self.model != PRIMARY_MODEL:
                raise
            logger.warning(f"{self.model} unavailable ({e}); falling back to {FALLBACK_MODEL}")
            self.model = FALLBACK_MODEL
            return self.client.messages.create(
                model=self.model,
                max_tokens=8096,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages,
            )

    def solve(
        self,
        issue_title: str,
        issue_body: str,
        available_tools: list[str] | None = None,
    ) -> str:
        tools_section = ""
        if available_tools:
            tools_section = (
                f"\n\nVerification binaries available on PATH: {', '.join(available_tools)}."
            )
        user_message = (
            f"# Issue: {issue_title}\n\n"
            f"{issue_body}{tools_section}\n\n"
            "Please fix this issue. Start by exploring the repository structure."
        )
        messages: list[dict] = [{"role": "user", "content": user_message}]
        iterations = 0

        while iterations < config.MAX_SOLVER_ITERATIONS:
            iterations += 1
            logger.info(f"Solver iteration {iterations} (model={self.model})")

            response = self._create(messages)
            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "end_turn":
                for block in response.content:
                    if hasattr(block, "text"):
                        return block.text
                return "Fix completed."

            if response.stop_reason != "tool_use":
                logger.warning(f"Unexpected stop reason: {response.stop_reason}")
                break

            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue

                logger.info(f"Tool: {block.name}({json.dumps(block.input)[:120]})")

                if block.name == "finish":
                    summary = block.input.get("summary", "Issue fixed.")
                    logger.info(f"Solver finished: {summary}")
                    tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": "Done."})
                    messages.append({"role": "user", "content": tool_results})
                    return summary

                result = self._dispatch(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result[:32000],
                })

            messages.append({"role": "user", "content": tool_results})
            _trim_old_tool_results(messages)

        raise RuntimeError(f"Solver hit max iterations ({config.MAX_SOLVER_ITERATIONS}) without finishing")

    def _dispatch(self, name: str, inp: dict) -> str:
        if name == "read_file":
            return self._read(inp["path"])
        if name == "write_file":
            return self._write(inp["path"], inp["content"])
        if name == "list_files":
            return self._list(inp.get("path", "."))
        if name == "search_code":
            return self._search(inp["pattern"], inp.get("path", "."))
        if name == "run_command":
            return self._run(inp["command"])
        return f"Unknown tool: {name}"

    def _read(self, rel_path: str) -> str:
        try:
            return (self.repo_path / rel_path).read_text(encoding="utf-8", errors="replace")
        except FileNotFoundError:
            return f"ERROR: File not found: {rel_path}"
        except Exception as e:
            return f"ERROR reading {rel_path}: {e}"

    def _write(self, rel_path: str, content: str) -> str:
        try:
            p = self.repo_path / rel_path
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            return f"Written: {rel_path}"
        except Exception as e:
            return f"ERROR writing {rel_path}: {e}"

    def _list(self, rel_path: str) -> str:
        p = self.repo_path / rel_path
        if not p.exists():
            return f"ERROR: Path does not exist: {rel_path}"
        entries = [f"{item.name}{'/' if item.is_dir() else ''}" for item in sorted(p.iterdir())]
        return "\n".join(entries) if entries else "(empty)"

    def _search(self, pattern: str, path: str) -> str:
        r = self.runner.exec(
            f"grep -rn -e {shlex.quote(pattern)} {shlex.quote(path)} 2>/dev/null | head -60"
        )
        return r["stdout"].strip() or f"No matches for '{pattern}' in {path}"

    def _run(self, command: str) -> str:
        r = self.runner.exec(command)
        parts = []
        if r["stdout"].strip():
            parts.append(r["stdout"].strip())
        if r["stderr"].strip():
            parts.append(f"[stderr]\n{r['stderr'].strip()}")
        parts.append(f"[exit code: {r['exit_code']}]")
        return "\n".join(parts)
