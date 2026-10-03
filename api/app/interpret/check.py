"""Explicit interpreter connection check (``POST /api/interpreter/check``).

``check_interpreter(settings)`` actively tests the configured backend and always returns an
:class:`~app.api.schemas.InterpreterCheck` (``ok: false`` + ``error`` instead of raising):

* ``openai_compat``: ``GET {base}/models`` (auth, reachability, ``latency_ms``, whether
  ``MIMIC_LLM_MODEL`` is listed), then one tiny chat completion with a synthetic, generated PNG
  (a blue rectangle, never user data) asking for a 2-field JSON answer. Reports
  ``supports_images`` (the rectangle was recognised) and which ``structured_output`` mode worked.
  ``latency_ms`` is the ``GET /models`` round trip (the chat probe's when /models is missing).
* ``claude_cli``: binary on PATH + ``claude --version``. No paid call unless ``deep=True``, which
  runs the real labeling flags once on the synthetic PNG in a temp dir.
* ``none``: ``ok: false``, ``not_configured``.

Error codes: ``not_configured``, ``unauthorized`` (401/403), ``unreachable`` (connect/DNS),
``timeout``, ``model_not_found``, ``no_image_support``, ``bad_response``. Messages never contain
the token.
"""

from __future__ import annotations

import base64
import shutil
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from app.api.schemas import InterpreterCheck, InterpreterCheckError
from app.config import Settings
from app.interpret.claude_cli import build_argv, parse_envelope, run_cli
from app.interpret.openai_compat import (
    StructuredMode,
    client_from_settings,
    image_part,
    missing_settings,
    modes_for,
)

CHECK_TIMEOUT_S = 30.0
VERSION_TIMEOUT_S = 15.0
DEEP_CLI_TIMEOUT_S = 90.0

PROBE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["shape", "color"],
    "properties": {
        "shape": {"type": "string", "enum": ["rectangle", "circle", "triangle", "none"]},
        "color": {"type": "string", "enum": ["red", "green", "blue", "other"]},
    },
}
PROBE_QUESTION = (
    "Look at the attached image. What is the main shape and its color? "
    'Answer with a JSON object like {"shape": "...", "color": "..."} using only these values: '
    "shape in rectangle|circle|triangle|none, color in red|green|blue|other."
)


def probe_png() -> bytes:
    """Synthetic test image: a solid blue rectangle on light gray (generated, no user data)."""
    import cv2
    import numpy as np

    img = np.full((96, 128, 3), 235, np.uint8)
    cv2.rectangle(img, (24, 24), (104, 72), (200, 90, 30), -1)  # BGR -> blue
    ok, buf = cv2.imencode(".png", img)
    if not ok:  # pragma: no cover - cv2 always encodes a plain array
        raise RuntimeError("could not encode probe image")
    return buf.tobytes()


def _now() -> datetime:
    return datetime.now(UTC)


def _result(
    mode: str, ok: bool, *, code: str | None = None, message: str | None = None, **kw: Any
) -> InterpreterCheck:
    err = InterpreterCheckError(code=code, message=message or code) if code else None  # type: ignore[arg-type]
    return InterpreterCheck(mode=mode, ok=ok, error=err, checked_at=_now(), **kw)  # type: ignore[arg-type]


def _answer_is_probe(payload: dict[str, Any] | None) -> bool:
    return bool(payload) and str(payload.get("shape", "")).lower() == "rectangle"


# ---- openai_compat -------------------------------------------------------------------------------


def _check_openai(settings: Settings, transport: httpx.BaseTransport | None) -> InterpreterCheck:
    mode = "openai_compat"
    missing = missing_settings(settings)
    if missing:
        return _result(
            mode, False, code="not_configured", message=f"missing: {', '.join(missing)}",
            model=settings.llm_model,
        )  # fmt: skip
    timeout = min(settings.llm_timeout_s, CHECK_TIMEOUT_S)
    client = client_from_settings(settings, transport=transport, timeout_s=timeout)
    assert client is not None  # missing_settings() covered every None case

    models = client.list_models()
    model_listed: bool | None = None
    latency: int | None = models.latency_ms
    if models.error_code in ("unauthorized", "unreachable", "timeout"):
        return _result(
            mode, False, code=models.error_code, message=f"GET /models: {models.message}",
            latency_ms=latency, model=client.model,
        )  # fmt: skip
    if models.ids is not None:
        model_listed = client.model in models.ids
    else:  # /models is optional on some gateways: report the chat probe's latency instead
        latency = None

    png_url = "data:image/png;base64," + base64.b64encode(probe_png()).decode("ascii")

    def build_messages(m: StructuredMode) -> list[dict[str, Any]]:
        return [
            {
                "role": "user",
                "content": [{"type": "text", "text": PROBE_QUESTION}, image_part(png_url)],
            }
        ]

    chat = client.chat_json(
        build_messages,
        PROBE_SCHEMA,
        schema_name="mimic_probe",
        modes=modes_for(settings.llm_supports_json_schema),
        deadline=time.monotonic() + timeout,
    )
    if latency is None:
        latency = chat.latency_ms
    common: dict[str, Any] = {
        "latency_ms": latency,
        "model": chat.model or client.model,
        "model_listed": model_listed,
    }
    if not chat.ok:
        code = chat.error_code or "bad_response"
        if model_listed is False and code == "bad_response":
            code = "model_not_found"
        supports = False if code == "no_image_support" else None
        return _result(
            mode, False, code=code, message=chat.message, supports_images=supports, **common
        )
    if not _answer_is_probe(chat.payload):
        return _result(
            mode, False, code="no_image_support",
            message="model answered but did not recognise the test image",
            supports_images=False, structured_output=chat.mode, **common,
        )  # fmt: skip
    return _result(mode, True, supports_images=True, structured_output=chat.mode, **common)


# ---- claude_cli ----------------------------------------------------------------------------------


def _check_claude(settings: Settings, deep: bool) -> InterpreterCheck:
    mode = "claude_cli"
    model = settings.claude_model
    if shutil.which(settings.claude_bin) is None:
        return _result(
            mode, False, code="not_configured",
            message="claude CLI not found on PATH (MIMIC_CLAUDE_BIN)", model=model,
        )  # fmt: skip
    with tempfile.TemporaryDirectory(prefix="mimic-check-") as tmp:
        tmp_dir = Path(tmp)
        run = run_cli([settings.claude_bin, "--version"], cwd=tmp_dir, timeout_s=VERSION_TIMEOUT_S)
        if run.spawn_error:
            return _result(mode, False, code="not_configured", message=run.spawn_error, model=model)
        if run.timed_out:
            return _result(
                mode, False, code="timeout", message="claude --version timed out", model=model
            )
        if run.returncode != 0 or not run.stdout.strip():
            return _result(
                mode, False, code="bad_response",
                message=f"claude --version exited with {run.returncode}", model=model,
            )  # fmt: skip
        if not deep:
            return _result(mode, True, latency_ms=run.duration_ms, model=model)

        (tmp_dir / "probe.png").write_bytes(probe_png())
        prompt = (
            f"Use the Read tool to open {tmp_dir.resolve() / 'probe.png'}. What is the main "
            "shape and its color?"
        )
        argv = build_argv(settings.claude_bin, model, prompt, PROBE_SCHEMA, tmp_dir.resolve())
        timeout = min(settings.claude_timeout_s, DEEP_CLI_TIMEOUT_S)
        deep_run = run_cli(argv, cwd=tmp_dir.resolve(), timeout_s=timeout)
    if deep_run.timed_out:
        return _result(mode, False, code="timeout", message="claude probe timed out", model=model)
    parsed = parse_envelope(deep_run.stdout)
    if deep_run.returncode != 0 or parsed.payload is None:
        reason = parsed.error or f"exit code {deep_run.returncode}"
        return _result(mode, False, code="bad_response", message=reason, model=model)
    ok = _answer_is_probe(parsed.payload)
    return _result(
        mode, ok, code=None if ok else "no_image_support",
        message=None if ok else "model did not recognise the test image",
        latency_ms=deep_run.duration_ms, model=model, supports_images=ok,
        structured_output="json_schema",
    )  # fmt: skip


# ---- entry point ---------------------------------------------------------------------------------


def check_interpreter(
    settings: Settings, *, deep: bool = False, transport: httpx.BaseTransport | None = None
) -> InterpreterCheck:
    """Actively test the configured interpreter. Never raises for backend failures.

    ``deep`` only affects ``claude_cli`` (one real, paid call). ``transport`` is for tests.
    """
    if settings.interpreter == "openai_compat":
        return _check_openai(settings, transport)
    if settings.interpreter == "claude_cli":
        return _check_claude(settings, deep)
    return _result(
        "none", False, code="not_configured", message="MIMIC_INTERPRETER=none (AI labeling off)"
    )
