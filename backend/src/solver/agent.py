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
5. Run the project's tests to verify your fix works
6. If tests fail, debug and iterate
7. When confident the fix is correct and tests pass, call `finish` with a clear summary

Rules:
- Only change what is necessary to fix the issue — do not refactor unrelated code
- Always run tests before calling `finish`
- If there are no tests, at least verify the code runs without errors
- Write clean, idiomatic code matching the project's style
- Do not add comments unless the logic is genuinely non-obvious

Handling unavailable toolchains:
- If dependency installation failed (e.g. node_modules missing, cargo not found),
  do not waste iterations trying to run compiler or test commands that require them.
  Make your best fix to the source code based on reading and understanding it, then
  call `finish` with an honest note that local verification was skipped due to a
  missing toolchain.
- Be decisive: if you have read the relevant code, understood the issue, and written
  a fix — call `finish`. Do not loop trying the same failing command repeatedly.
"""


class IssueSolver:
    def __init__(self, runner: NativeRunner, repo_path: Path):
        self.client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        self.runner = runner
        self.repo_path = repo_path

    def solve(self, issue_title: str, issue_body: str, setup_warnings: list[str] | None = None) -> str:
        setup_section = ""
        if setup_warnings:
            joined = "\n".join(f"- {w}" for w in setup_warnings)
            setup_section = (
                f"\n\n**Setup warnings (dependency installation had issues):**\n{joined}\n"
                "Some build tools may be unavailable. Fix the source code directly "
                "and skip compilation checks if they fail."
            )
        user_message = (
            f"# Issue: {issue_title}\n\n"
            f"{issue_body}"
            f"{setup_section}\n\n"
            "Please fix this issue. Start by exploring the repository structure."
        )
        messages: list[dict] = [{"role": "user", "content": user_message}]
        iterations = 0

        while iterations < config.MAX_SOLVER_ITERATIONS:
            iterations += 1
            logger.info(f"Solver iteration {iterations}")

            response = self.client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=8096,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages,
            )
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
                    "content": result[:8000],
                })

            messages.append({"role": "user", "content": tool_results})

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
