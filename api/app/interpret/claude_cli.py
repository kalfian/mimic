"""Layer B via Claude Code headless (``claude -p``) as a subprocess (PLAN §7.2, D2).

CLI spike — Claude Code 2.1.287, 2026-10-02 (3 live calls, ``--model sonnet``)
==============================================================================

Flags (from ``claude --help`` on 2.1.287):

* ``-p/--print <prompt>``, ``--output-format json`` (single result object on stdout).
* ``--json-schema <schema>`` takes the schema as an **inline JSON string** (the help example is
  inline; a path is not documented). Verified: inline schema with ``enum`` / ``required`` /
  ``additionalProperties: false`` is honoured.
* ``--model sonnet`` (alias) resolved to ``claude-sonnet-5-5`` (key of ``modelUsage``).
* ``--permission-mode`` choices are ``acceptEdits | auto | bypassPermissions | manual |
  dontAsk | plan``. **There is no ``default`` mode** (PLAN §7.2 said ``default``) -> we use
  ``dontAsk``: anything not pre-approved by ``--allowedTools`` is denied, never prompted.
* ``--tools Read`` restricts the *available* built-in tool set (stronger than
  ``--allowedTools``, which only pre-approves). With ``--tools Read`` the model reported its
  only tools as ``Read`` + ``StructuredOutput``; asked to create a file and run ``touch`` it
  could not (no file appeared). We pass both ``--tools Read`` and ``--allowedTools Read``.
* Isolation extras, all verified to work with OAuth auth from ``$HOME``:
  ``--safe-mode`` (no user CLAUDE.md / hooks / skills / plugins / MCP servers),
  ``--strict-mcp-config`` (no MCP servers without ``--mcp-config``),
  ``--no-session-persistence`` (nothing about the job is saved as a resumable session).
  ``--bare`` is NOT usable: it requires ``ANTHROPIC_API_KEY`` (OAuth ignored).
* ``--add-dir <dir>`` + ``cwd=<dir>``: ``Read`` on an absolute PNG path inside the dir works
  (a synthetic blue rectangle was identified as ``rectangle`` / ``blue``).

Envelope (stdout, exit 0), relevant keys::

    {"type": "result", "subtype": "success", "is_error": false, "num_turns": 2,
     "result": "{\\"answer\\":\\"ok\\"}",          # the same JSON, as a string
     "structured_output": {"answer": "ok"},        # <- parsed object when --json-schema is used
     "stop_reason": "tool_use", "terminal_reason": "completed",
     "permission_denials": [], "duration_ms": 1927, "total_cost_usd": 0.012,
     "modelUsage": {"claude-sonnet-5-5": {...}}, "session_id": "...", "usage": {...}, ...}

Structured output is delivered through an internal ``StructuredOutput`` tool call, hence
``stop_reason: "tool_use"`` and ``num_turns >= 2`` on success. Latency: 2-5 s for tiny calls.

Defensive parser order (PLAN §7.2): (1) ``structured_output`` dict, (2) ``json.loads(result)``,
(3) first balanced ``{...}`` in ``result``. Non-zero exit, ``is_error: true`` or a non-success
``subtype`` -> ``error``. Every failure returns the deterministic fallback (``fallback.py``)
with ``status`` = ``timeout`` / ``error``; nothing here raises for expected failures.

Timeout: the CLI runs in its own session/process group (``start_new_session=True``); on
timeout the whole group gets SIGTERM, then SIGKILL after a 3 s grace, so helper processes
cannot outlive the job. Worst case wall time = timeout + ~4.5 s.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from app.config import Settings
from app.interpret.base import InterpretationInput, InterpretationResult
from app.interpret.fallback import build_fallback
from app.interpret.payload import first_balanced_object, sanitize_payload, write_raw
from app.interpret.prompt import build_prompt, select_images
from app.interpret.schema import InterpretationPayload, build_schema
from app.models.ir import InterpreterProvider

KILL_GRACE_S = 3.0
PROGRESS_TICK_S = 1.0
STDERR_TAIL_BYTES = 2048
STDOUT_RAW_MAX = 64 * 1024

ProgressCallback = Callable[[float], None]


# ---- subprocess ----------------------------------------------------------------------------------


@dataclass(slots=True)
class CliRun:
    """Outcome of one CLI process run."""

    returncode: int | None
    stdout: str
    stderr: str
    timed_out: bool
    duration_ms: int
    spawn_error: str | None = None


def build_argv(
    claude_bin: str, model: str, prompt: str, schema: dict[str, Any], keyframes_dir: Path
) -> list[str]:
    """Argv for one labeling call. Read-only tools, no prompts, no session saved."""
    return [
        claude_bin,
        "-p",
        prompt,
        "--output-format",
        "json",
        "--json-schema",
        json.dumps(schema, separators=(",", ":")),
        "--model",
        model,
        "--tools",
        "Read",
        "--allowedTools",
        "Read",
        "--add-dir",
        str(keyframes_dir),
        "--permission-mode",
        "dontAsk",
        "--strict-mcp-config",
        "--no-session-persistence",
        "--safe-mode",
    ]


def _signal_group(pgid: int, sig: signal.Signals) -> None:
    try:
        os.killpg(pgid, sig)
    except (ProcessLookupError, PermissionError):
        pass


def _kill_group(proc: subprocess.Popen[bytes], grace_s: float) -> None:
    """SIGTERM the process group, wait ``grace_s``, then SIGKILL whatever is left."""
    pgid = proc.pid  # start_new_session=True -> the child leads its own group
    _signal_group(pgid, signal.SIGTERM)
    try:
        proc.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        pass
    _signal_group(pgid, signal.SIGKILL)  # also sweeps grandchildren if the leader already exited
    try:
        proc.wait(timeout=0.5)
    except subprocess.TimeoutExpired:
        pass


def run_cli(
    argv: list[str],
    *,
    cwd: Path,
    timeout_s: float,
    on_progress: ProgressCallback | None = None,
    grace_s: float = KILL_GRACE_S,
) -> CliRun:
    """Run the CLI with a hard timeout that kills the whole process group."""
    start = time.monotonic()

    def elapsed_ms() -> int:
        return int((time.monotonic() - start) * 1000)

    try:
        proc = subprocess.Popen(  # noqa: S603 - argv is built by us, no shell
            argv,
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            env=os.environ.copy(),  # the CLI needs its own auth from $HOME
        )
    except OSError as e:
        reason = f"{type(e).__name__}: {e.strerror or e}"
        return CliRun(None, "", "", False, elapsed_ms(), spawn_error=reason)

    deadline = start + timeout_s
    out_b, err_b = b"", b""
    timed_out = False
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            timed_out = True
            break
        try:
            out_b, err_b = proc.communicate(timeout=min(PROGRESS_TICK_S, remaining))
            break
        except subprocess.TimeoutExpired:
            if on_progress is not None:
                try:
                    on_progress(min((time.monotonic() - start) / timeout_s, 1.0))
                except Exception:  # noqa: BLE001 - progress must never break the call
                    pass

    if timed_out:
        _kill_group(proc, grace_s)
        try:
            out_b, err_b = proc.communicate(timeout=0.5)
        except subprocess.TimeoutExpired:  # a re-sessioned grandchild still holds the pipes
            proc.kill()
            for stream in (proc.stdout, proc.stderr):
                if stream is not None:
                    stream.close()
            out_b, err_b = b"", b""

    return CliRun(
        returncode=proc.returncode,
        stdout=(out_b or b"").decode("utf-8", errors="replace"),
        stderr=(err_b or b"").decode("utf-8", errors="replace"),
        timed_out=timed_out,
        duration_ms=elapsed_ms(),
    )


# ---- envelope parsing ----------------------------------------------------------------------------


@dataclass(slots=True)
class ParseOutcome:
    envelope: dict[str, Any] | None
    payload: dict[str, Any] | None
    path: str  # structured_output | result_json | result_braces | no_envelope | no_payload | ...
    error: str | None = None


def _load_envelope(stdout: str) -> dict[str, Any] | None:
    text = stdout.strip()
    if not text:
        return None
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        obj = None
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, list):  # stream-style array: take the last result object
        for item in reversed(obj):
            if isinstance(item, dict) and item.get("type") == "result":
                return item
        return None
    for line in reversed(text.splitlines()):  # noise before the JSON line
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            cand = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(cand, dict):
            return cand
    return None


def parse_envelope(stdout: str) -> ParseOutcome:
    """Find the structured payload in the CLI's JSON envelope (order in module docstring)."""
    env = _load_envelope(stdout)
    if env is None:
        return ParseOutcome(None, None, "no_envelope", "stdout is not a JSON envelope")
    if env.get("is_error") is True:
        return ParseOutcome(env, None, "is_error", f"envelope is_error ({env.get('subtype')})")
    subtype = env.get("subtype")
    if subtype is not None and subtype != "success":
        return ParseOutcome(env, None, "subtype", f"envelope subtype {subtype!r}")

    so = env.get("structured_output")
    if isinstance(so, dict):
        return ParseOutcome(env, so, "structured_output")
    result = env.get("result")
    if isinstance(result, dict):
        return ParseOutcome(env, result, "result_json")
    if isinstance(result, str):
        try:
            obj = json.loads(result)
        except json.JSONDecodeError:
            obj = None
        if isinstance(obj, dict):
            return ParseOutcome(env, obj, "result_json")
        obj = first_balanced_object(result)
        if obj is not None:
            return ParseOutcome(env, obj, "result_braces")
    return ParseOutcome(env, None, "no_payload", "no JSON object in envelope")


# ---- interpreter ---------------------------------------------------------------------------------


@dataclass(slots=True)
class _Raw:
    """What goes into ``interpretation_raw.json`` (no secrets, no environment)."""

    status: str
    parse_path: str | None = None
    error: str | None = None
    exit_code: int | None = None
    timed_out: bool = False
    duration_ms: int | None = None
    argv: list[str] = field(default_factory=list)
    envelope: dict[str, Any] | None = None
    stdout_text: str | None = None
    stderr_tail: str = ""
    issues: list[str] = field(default_factory=list)


def _redacted_argv(argv: list[str]) -> list[str]:
    out = list(argv)
    for flag, label in (("-p", "prompt"), ("--json-schema", "schema")):
        if flag in out:
            i = out.index(flag) + 1
            if i < len(out):
                out[i] = f"<{label}: {len(out[i])} chars>"
    if out:
        out[0] = Path(out[0]).name
    return out


def _model_from_envelope(env: dict[str, Any] | None, default: str) -> str:
    usage = env.get("modelUsage") if env else None
    if isinstance(usage, dict) and usage:
        return str(next(iter(usage)))
    return default


class ClaudeCliInterpreter:
    """Labels elements with ``claude -p``; any failure -> fallback with timeout/error status."""

    provider: InterpreterProvider = "claude_cli"

    def __init__(self, settings: Settings, on_progress: ProgressCallback | None = None):
        self.claude_bin = settings.claude_bin
        self.model = settings.claude_model
        self.timeout_s = settings.claude_timeout_s
        self.on_progress = on_progress

    # Kept small and explicit so every exit path writes the raw record and returns a result.
    def interpret(self, inp: InterpretationInput) -> InterpretationResult:
        t0 = time.monotonic()
        keyframes_dir = (inp.job_dir / "keyframes").resolve()

        def fail(status: str, reason: str, raw: _Raw) -> InterpretationResult:
            ms = int((time.monotonic() - t0) * 1000)
            raw.status, raw.error, raw.duration_ms = status, reason, ms
            self._write_raw(inp.job_dir, raw)
            return build_fallback(
                inp,
                status=status,  # type: ignore[arg-type]
                provider=self.provider,
                model=self.model,
                duration_ms=ms,
                notes=[reason],
            )

        images = select_images(inp, keyframes_dir) if keyframes_dir.is_dir() else []
        if not inp.elements:
            return fail("fallback", "no elements to label", _Raw(status="fallback"))
        if not images:
            return fail("fallback", "no keyframes available to label", _Raw(status="fallback"))

        schema = build_schema([e.id for e in inp.elements])
        argv = build_argv(
            self.claude_bin, self.model, build_prompt(inp, images), schema, keyframes_dir
        )
        raw = _Raw(status="running", argv=_redacted_argv(argv))

        run = run_cli(
            argv, cwd=keyframes_dir, timeout_s=self.timeout_s, on_progress=self.on_progress
        )
        raw.exit_code, raw.timed_out = run.returncode, run.timed_out
        raw.stderr_tail = run.stderr.encode("utf-8")[-STDERR_TAIL_BYTES:].decode("utf-8", "ignore")

        if run.spawn_error is not None:
            return fail("error", f"could not start claude ({run.spawn_error})", raw)
        if run.timed_out:
            return fail("timeout", f"claude timed out after {self.timeout_s:g} s", raw)

        parsed = parse_envelope(run.stdout)
        raw.parse_path, raw.envelope = parsed.path, parsed.envelope
        if parsed.envelope is None:
            raw.stdout_text = run.stdout[:STDOUT_RAW_MAX]
        if run.returncode != 0:
            return fail("error", f"claude exited with code {run.returncode}", raw)
        if parsed.payload is None:
            return fail("error", parsed.error or "unparseable output", raw)

        try:
            payload = InterpretationPayload.model_validate(parsed.payload)
        except ValidationError as e:
            return fail("error", f"payload failed validation ({e.error_count()} errors)", raw)

        model_used = _model_from_envelope(parsed.envelope, self.model)
        base = build_fallback(inp, status="fallback", provider=self.provider)
        ms = int((time.monotonic() - t0) * 1000)
        result, issues = sanitize_payload(
            payload, inp, base, provider=self.provider, model=model_used, duration_ms=ms
        )
        raw.issues = issues
        if result is None:
            return fail("error", "AI output had no valid element labels", raw)
        raw.status, raw.duration_ms = "ok", ms
        self._write_raw(inp.job_dir, raw)
        return result

    @staticmethod
    def _write_raw(job_dir: Path, raw: _Raw) -> None:
        write_raw(
            job_dir,
            {
                "provider": "claude_cli",
                "status": raw.status,
                "parse_path": raw.parse_path,
                "error": raw.error,
                "exit_code": raw.exit_code,
                "timed_out": raw.timed_out,
                "duration_ms": raw.duration_ms,
                "argv": raw.argv,
                "issues": raw.issues,
                "stderr_tail": raw.stderr_tail,
                "envelope": raw.envelope,
                "stdout_text": raw.stdout_text,
            },
        )
