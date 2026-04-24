import json
import logging
import shlex
import time
from pathlib import Path

import google.genai as genai
import google.genai.types as genai_types

import config
from src.docker.manager import NativeRunner

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Gemini client — reuses the ANTHROPIC_API_KEY env var as the Gemini API key
# so callers only need to change the key value, not the variable name.
# ---------------------------------------------------------------------------
_gemini_client = genai.Client(api_key=config.ANTHROPIC_API_KEY)

# Transient API failures we retry with backoff rather than letting them blow
# up the whole pipeline (which would discard iterations of solver state).
_RETRYABLE_API_ERRORS: tuple[type[Exception], ...] = (
    Exception,   # google-genai surfaces transient errors as generic exceptions;
                 # we filter by message in _create() for permanent ones.
)

_API_RETRY_BACKOFF_SECONDS: tuple[int, ...] = (30, 60, 120)

# ---------------------------------------------------------------------------
# Model cascade: most capable → lighter fallbacks
# ---------------------------------------------------------------------------
PRIMARY_MODEL   = "gemini-3.1-pro-preview"          # most powerful (3.1 Pro)
FALLBACK_MODELS = [
    "gemini-3.1-pro-preview-customtools",            # same model, better tool adherence
    "gemini-3-flash-preview",                        # fast, Pro-grade reasoning
    "gemini-3.1-flash-lite-preview",                 # cheapest/fastest Gemini 3
    "gemini-2.5-pro-exp-03-25",                      # last resort: proven 2.5 Pro
    "gemini-2.5-flash",                              # reliable 2.5 workhorse
]
ALL_MODELS = [PRIMARY_MODEL] + FALLBACK_MODELS

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

# ---------------------------------------------------------------------------
# Convert our Anthropic-style tool schemas to Gemini FunctionDeclarations
# ---------------------------------------------------------------------------
def _build_gemini_tools() -> list[genai_types.Tool]:
    declarations = []
    for t in TOOLS:
        schema = t["input_schema"]
        props = {}
        for prop_name, prop_def in schema.get("properties", {}).items():
            props[prop_name] = genai_types.Schema(
                type=genai_types.Type.STRING,
                description=prop_def.get("description", ""),
            )
        fn = genai_types.FunctionDeclaration(
            name=t["name"],
            description=t["description"],
            parameters=genai_types.Schema(
                type=genai_types.Type.OBJECT,
                properties=props,
                required=schema.get("required", []),
            ),
        )
        declarations.append(fn)
    return [genai_types.Tool(function_declarations=declarations)]


_GEMINI_TOOLS = _build_gemini_tools()

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

Scope — critical:
- You are responsible ONLY for failures caused by your own changes. If verification surfaces failures in files/tests unrelated to what you modified, those are PRE-EXISTING issues in the repo. Do not try to fix them. Call `finish` as soon as your own changes are green, noting any pre-existing failures in the summary so a human can triage them separately.
- Chasing every red test in a large repo is how the solver hits max iterations. Stay scoped: fix what you broke, ignore what was already broken.
"""

# Keep the last N tool-result turns verbatim; older ones are shrunk to a stub.
TOOL_RESULT_WINDOW = 25
TOOL_RESULT_STUB_LEN = 120


# ---------------------------------------------------------------------------
# Message format helpers
# ---------------------------------------------------------------------------
def _to_gemini_contents(messages: list[dict]) -> list[genai_types.Content]:
    """Convert our internal message list to Gemini Content objects."""
    contents = []
    for m in messages:
        role = "user" if m["role"] == "user" else "model"
        raw = m["content"]

        # Plain string (initial user prompt)
        if isinstance(raw, str):
            contents.append(genai_types.Content(role=role, parts=[genai_types.Part(text=raw)]))
            continue

        # List of blocks (tool_use / tool_result / text)
        parts = []
        for block in raw:
            if isinstance(block, str):
                parts.append(genai_types.Part(text=block))
                continue
            if not isinstance(block, dict):
                # Gemini SDK response objects (assistant turn) — convert via their dict repr
                block_dict = block if isinstance(block, dict) else vars(block)
                _extract_parts_from_sdk_block(block_dict, parts)
                continue

            btype = block.get("type")
            if btype == "tool_result":
                parts.append(
                    genai_types.Part(
                        function_response=genai_types.FunctionResponse(
                            name=block.get("name", block.get("tool_use_id", "unknown")),
                            response={"result": block.get("content", "")},
                        )
                    )
                )
            elif btype == "tool_use":
                parts.append(
                    genai_types.Part(
                        function_call=genai_types.FunctionCall(
                            name=block["name"],
                            args=block.get("input", {}),
                        )
                    )
                )
            elif btype == "text":
                parts.append(genai_types.Part(text=block.get("text", "")))

        if parts:
            contents.append(genai_types.Content(role=role, parts=parts))

    return contents


def _extract_parts_from_sdk_block(block: dict, parts: list) -> None:
    """Handle Gemini SDK response Content/Part objects stored in messages."""
    if "text" in block:
        parts.append(genai_types.Part(text=block["text"]))
    elif "function_call" in block:
        fc = block["function_call"]
        parts.append(
            genai_types.Part(
                function_call=genai_types.FunctionCall(
                    name=fc.get("name", ""), args=fc.get("args", {})
                )
            )
        )


def _is_permanent_error(e: Exception) -> bool:
    """Return True for errors that should NOT be retried (auth, quota exhausted, bad request)."""
    msg = str(e).lower()
    permanent_keywords = ("api_key", "invalid", "permission", "not found", "quota", "billing")
    return any(k in msg for k in permanent_keywords)


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
        self.runner = runner
        self.repo_path = repo_path
        self._model_index = 0  # index into ALL_MODELS
        # Conversation state persists across solve() / continue_after_verification()
        self.messages: list[dict] = []

    @property
    def model(self) -> str:
        return ALL_MODELS[self._model_index]

    def _next_model(self) -> bool:
        """Advance to the next fallback model. Returns False if exhausted."""
        if self._model_index + 1 >= len(ALL_MODELS):
            return False
        self._model_index += 1
        logger.warning(f"Falling back to model: {self.model}")
        return True

    def _create(self, messages: list[dict]) -> genai_types.GenerateContentResponse:
        contents = _to_gemini_contents(messages)
        config_obj = genai_types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            tools=_GEMINI_TOOLS,
            max_output_tokens=8096,
            temperature=1.0,
        )

        attempt = 0
        while True:
            try:
                return _gemini_client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=config_obj,
                )
            except Exception as e:
                if _is_permanent_error(e):
                    raise

                # Model-level failure (404, unavailable) → try next model first
                msg = str(e).lower()
                if any(k in msg for k in ("not found", "unavailable", "deprecated")):
                    if self._next_model():
                        attempt = 0
                        continue
                    raise

                # Transient error → backoff / retry
                if attempt >= len(_API_RETRY_BACKOFF_SECONDS):
                    if self._next_model():
                        attempt = 0
                        continue
                    raise
                wait = _API_RETRY_BACKOFF_SECONDS[attempt]
                attempt += 1
                logger.warning(
                    f"API error ({type(e).__name__}) on attempt {attempt}/"
                    f"{len(_API_RETRY_BACKOFF_SECONDS)}: {e}. Retrying in {wait}s..."
                )
                time.sleep(wait)

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
        self.messages = [{"role": "user", "content": user_message}]
        return self._run_loop()

    def continue_after_verification(self, verification_error: str) -> str:
        """Continue from existing conversation after the pipeline's post-finish
        verification gate caught failures."""
        if not self.messages:
            raise RuntimeError("continue_after_verification called before solve()")

        diff = self.runner._run("git diff --name-only HEAD 2>&1", timeout=30)
        modified_files = [
            line.strip() for line in (diff.get("stdout") or "").splitlines() if line.strip()
        ]
        modified_list = "\n".join(f"  - {f}" for f in modified_files) or "  (none detected)"

        self.messages.append({
            "role": "user",
            "content": (
                "The code you called `finish` on did NOT pass the pipeline's "
                "post-finish verification gate (lint, typecheck, tests).\n\n"
                "Files YOU modified in this pass:\n"
                f"{modified_list}\n\n"
                "Scope rules — read carefully:\n"
                "- Fix ONLY failures that are caused by your changes. A failure "
                "is yours if it's in one of the modified files above, in a test "
                "that imports one of those files, or in code that calls APIs "
                "you introduced.\n"
                "- Failures in completely unrelated files/tests are PRE-EXISTING "
                "issues in the repository, NOT caused by your change. Do NOT "
                "try to fix them. Leave them alone and call `finish` as soon as "
                "your own changes pass.\n"
                "- If every remaining failure is pre-existing, call `finish` "
                "immediately with a summary noting which failures were pre-existing "
                "and skipped.\n"
                "- Chasing every red test in the repo is how the solver times "
                "out. Stay scoped.\n\n"
                "Verification failures:\n\n"
                f"{verification_error}"
            ),
        })
        return self._run_loop(max_seconds=300)

    def _run_loop(self, max_seconds: int | None = None) -> str:
        iterations = 0
        deadline = time.monotonic() + max_seconds if max_seconds else None

        while iterations < config.MAX_SOLVER_ITERATIONS:
            if deadline is not None and time.monotonic() >= deadline:
                logger.warning(
                    f"Solver wall-clock budget ({max_seconds}s) exceeded at "
                    f"iteration {iterations}; returning partial fix"
                )
                return (
                    "(partial) Time budget exceeded while addressing verification "
                    "failures — current changes will be pushed for CI to evaluate."
                )
            iterations += 1
            logger.info(f"Solver iteration {iterations} (model={self.model})")

            response = self._create(self.messages)
            candidate = response.candidates[0]
            content = candidate.content  # genai_types.Content

            # Store assistant turn as a list of serialisable dicts
            assistant_blocks = []
            for part in content.parts:
                if part.text:
                    assistant_blocks.append({"type": "text", "text": part.text})
                elif part.function_call:
                    assistant_blocks.append({
                        "type": "tool_use",
                        "name": part.function_call.name,
                        "input": dict(part.function_call.args),
                        # Gemini has no per-call ID; use name as stable key
                        "id": part.function_call.name,
                    })
            self.messages.append({"role": "assistant", "content": assistant_blocks})

            # Check finish reason
            finish_reason = candidate.finish_reason
            has_function_calls = any(p.function_call for p in content.parts)

            if not has_function_calls:
                # Model responded with text only → treat as end_turn
                for block in assistant_blocks:
                    if block.get("type") == "text" and block.get("text"):
                        return block["text"]
                return "Fix completed."

            # Process tool calls
            tool_results = []
            for part in content.parts:
                if not part.function_call:
                    continue

                fc = part.function_call
                name = fc.name
                inp = dict(fc.args)

                logger.info(f"Tool: {name}({json.dumps(inp)[:120]})")

                if name == "finish":
                    summary = inp.get("summary", "Issue fixed.")
                    logger.info(f"Solver finished: {summary}")
                    tool_results.append({
                        "type": "tool_result",
         tart by exploring the repository structure."
        )
        self.messages = [{"role": "user", "content": user_message}]
        return self._run_loop()

    def continue_after_verification(self, verification_error: str) -> str:
        """Continue from existing conversation after the pipeline's post-finish
        verification gate caught failures. Preserves every tool call the solver
        already made so it doesn't re-read the same files from scratch."""
        if not self.messages:
            raise RuntimeError("continue_after_verification called before solve()")

        # List the files THIS solver pass actually changed. Used in the prompt
        # below to keep the solver from chasing pre-existing brokenness that
        # belongs to other commits.
        diff = self.runner._run("git diff --name-only HEAD 2>&1", timeout=30)
        modified_files = [
            line.strip() for line in (diff.get("stdout") or "").splitlines() if line.strip()
        ]
        modified_list = "\n".join(f"  - {f}" for f in modified_files) or "  (none detected)"

        self.messages.append({
            "role": "user",
            "content": (
                "The code you called `finish` on did NOT pass the pipeline's "
                "post-finish verification gate (lint, typecheck, tests).\n\n"
                "Files YOU modified in this pass:\n"
                f"{modified_list}\n\n"
                "Scope rules — read carefully:\n"
                "- Fix ONLY failures that are caused by your changes. A failure "
                "is yours if it's in one of the modified files above, in a test "
                "that imports one of those files, or in code that calls APIs "
                "you introduced.\n"
                "- Failures in completely unrelated files/tests are PRE-EXISTING "
                "issues in the repository, NOT caused by your change. Do NOT "
                "try to fix them. Leave them alone and call `finish` as soon as "
                "your own changes pass.\n"
                "- If every remaining failure is pre-existing, call `finish` "
                "immediately with a summary noting which failures were pre-existing "
                "and skipped.\n"
                "- Chasing every red test in the repo is how the solver times "
                "out. Stay scoped.\n\n"
                "Verification failures:\n\n"
                f"{verification_error}"
            ),
        })
        return self._run_loop(max_seconds=300)

    def _run_loop(self, max_seconds: int | None = None) -> str:
        iterations = 0
        deadline = time.monotonic() + max_seconds if max_seconds else None
        while iterations < config.MAX_SOLVER_ITERATIONS:
            if deadline is not None and time.monotonic() >= deadline:
                logger.warning(
                    f"Solver wall-clock budget ({max_seconds}s) exceeded at "
                    f"iteration {iterations}; returning partial fix"
                )
                return (
                    "(partial) Time budget exceeded while addressing verification "
                    "failures — current changes will be pushed for CI to evaluate."
                )
            iterations += 1
            logger.info(f"Solver iteration {iterations} (model={self.model})")

            response = self._create(self.messages)
            self.messages.append({"role": "assistant", "content": response.content})

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
                    self.messages.append({"role": "user", "content": tool_results})
                    return summary

                result = self._dispatch(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result[:32000],
                })

            self.messages.append({"role": "user", "content": tool_results})
            _trim_old_tool_results(self.messages)

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
