"""Layer B via any OpenAI-compatible ``/v1`` gateway (``MIMIC_INTERPRETER=openai_compat``).

``POST {MIMIC_LLM_BASE_URL}/chat/completions`` with the job's keyframes attached as base64
``image_url`` data URLs, the same prompt as ``claude_cli`` and the same per-job schema (closed
element-id enums, no numeric fields except confidences — D4).

Structured output, best first (``MIMIC_LLM_SUPPORTS_JSON_SCHEMA``):

* ``auto`` (default): ``json_schema`` (strict) -> ``json_object`` -> ``prompt_only``. The next
  mode is tried only when the gateway answers 400/422 with a body that points at
  ``response_format`` / schema support. The mode that worked is cached per (base URL, model)
  for the life of the process.
* ``true``: ``json_schema`` only. ``false``: ``json_object`` -> ``prompt_only``.

For ``json_object`` / ``prompt_only`` the schema is spelled out in the prompt. Answers are
always parsed defensively (whole text, code fences, first balanced ``{...}``) and repaired by
:func:`app.interpret.payload.sanitize_payload`, exactly like the CLI path. Any failure returns
the deterministic fallback with ``status`` ``timeout`` / ``error``; nothing raises.

Secrets: the bearer token is only ever placed in the ``Authorization`` header. It is redacted
from every message, the raw record and ``repr``; request headers and bodies are never stored.

Timeouts: ``MIMIC_LLM_TIMEOUT_S`` is the budget for the whole interpretation; each attempt gets
the remaining budget as its httpx timeout (httpx timeouts apply per network operation, which for
a non-streamed completion is effectively the full response wait).
"""

from __future__ import annotations

import base64
import copy
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import SecretStr, ValidationError

from app.config import Settings
from app.interpret.base import InterpretationInput, InterpretationResult
from app.interpret.fallback import build_fallback
from app.interpret.payload import REDACTED, extract_json_object, redact, sanitize_payload, write_raw
from app.interpret.prompt import SYSTEM_PROMPT, build_prompt, select_images
from app.interpret.schema import InterpretationPayload, build_schema
from app.models.ir import InterpreterProvider

StructuredMode = Literal["json_schema", "json_object", "prompt_only"]
ErrorCode = Literal[
    "unauthorized", "unreachable", "timeout", "model_not_found", "no_image_support", "bad_response"
]
ProgressCallback = Callable[[float], None]

ALL_MODES: tuple[StructuredMode, ...] = ("json_schema", "json_object", "prompt_only")
CONTEXT_IMAGE_MAX_SIDE = 1280
ELEMENT_IMAGE_MAX_SIDE = 800
JPEG_QUALITY = 85
BODY_EXCERPT = 200
RESPONSE_RAW_MAX = 64 * 1024
CONNECT_TIMEOUT_S = 10.0

#: Working structured-output mode per (base_url, model), learned at runtime.
_MODE_CACHE: dict[tuple[str, str], StructuredMode] = {}

_MODE_HINTS = ("response_format", "json_schema", "json_object", "json mode", "structured", "schema")
_GENERIC_UNSUPPORTED = ("not supported", "unsupported", "not support")
_IMAGE_HINTS = ("image", "vision", "multimodal")


def modes_for(setting: bool | Literal["auto"]) -> list[StructuredMode]:
    if setting is True:
        return ["json_schema"]
    if setting is False:
        return ["json_object", "prompt_only"]
    return list(ALL_MODES)


def to_strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Variant for ``strict: true``: every property required (optional ones nullable), no
    ``maxLength`` (lengths are clipped by the sanitizer anyway)."""

    def walk(node: Any) -> Any:
        if isinstance(node, list):
            return [walk(n) for n in node]
        if not isinstance(node, dict):
            return node
        node = {k: walk(v) for k, v in node.items() if k != "maxLength"}
        props = node.get("properties")
        if node.get("type") == "object" and isinstance(props, dict):
            required = set(node.get("required", []))
            for key, sub in props.items():
                if key not in required and isinstance(sub, dict):
                    t = sub.get("type")
                    if isinstance(t, str) and t != "null":
                        sub["type"] = [t, "null"]
                    if "enum" in sub and None not in sub["enum"]:
                        sub["enum"] = [*sub["enum"], None]
            node["required"] = list(props)
            node["additionalProperties"] = False
        return node

    return walk(copy.deepcopy(schema))


def _strip_nulls(obj: Any) -> Any:
    """Strict-mode answers carry ``null`` for omitted optionals; the payload model wants absence
    for list fields, so drop top-level null lists."""
    if isinstance(obj, dict):
        return {k: v for k, v in obj.items() if not (k in ("structure", "notes") and v is None)}
    return obj


# ---- image encoding ------------------------------------------------------------------------------


def encode_image(path: Path, max_side: int) -> str | None:
    """File -> ``data:image/jpeg;base64,...`` (downscaled), PNG passthrough if cv2 fails."""
    try:
        import cv2

        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is not None and img.size > 0:
            h, w = img.shape[:2]
            scale = max_side / max(h, w)
            if scale < 1.0:
                size = (max(1, round(w * scale)), max(1, round(h * scale)))
                img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if ok:
                return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")
    except Exception:  # noqa: BLE001 - fall through to raw bytes
        pass
    try:
        return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")
    except OSError:
        return None


def image_part(data_url: str) -> dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": data_url}}


# ---- HTTP client ---------------------------------------------------------------------------------


@dataclass(slots=True)
class ChatOutcome:
    ok: bool
    mode: StructuredMode | None = None
    payload: dict[str, Any] | None = None
    parse_path: str | None = None
    model: str | None = None
    error_code: ErrorCode | None = None
    message: str | None = None
    attempts: list[dict[str, Any]] = field(default_factory=list)
    response: dict[str, Any] | None = None
    latency_ms: int = 0


@dataclass(slots=True)
class ModelsOutcome:
    ids: list[str] | None
    http_status: int | None
    latency_ms: int
    error_code: ErrorCode | None = None
    message: str | None = None


def _excerpt(text: str) -> str:
    return " ".join(text.split())[:BODY_EXCERPT]


def _classify_http(status: int, body: str) -> tuple[ErrorCode, str]:
    low = body.lower()
    if status in (401, 403):
        return "unauthorized", f"HTTP {status}: credentials rejected"
    if status in (408, 504):
        return "timeout", f"HTTP {status}: gateway timeout"
    if "model" in low and any(
        h in low
        for h in ("not found", "does not exist", "unknown model", "invalid model", "no such")
    ):
        return "model_not_found", f"HTTP {status}: {_excerpt(body)}"
    if status in (400, 415, 422) and ("image" in low or "vision" in low or "multimodal" in low):
        return "no_image_support", f"HTTP {status}: {_excerpt(body)}"
    if status == 404:
        return "bad_response", "HTTP 404: endpoint not found (check MIMIC_LLM_BASE_URL)"
    return "bad_response", f"HTTP {status}: {_excerpt(body)}"


def _is_unsupported_mode(status: int, body: str) -> bool:
    """400/422 that rejects the ``response_format`` (not images, not the model)."""
    if status not in (400, 422):
        return False
    low = body.lower()
    if any(h in low for h in _MODE_HINTS):
        return True
    if any(h in low for h in _IMAGE_HINTS) or "model" in low:
        return False
    return any(h in low for h in _GENERIC_UNSUPPORTED)


def _message_content(data: Any) -> str | None:
    try:
        msg = data["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return None
    if not isinstance(msg, dict):
        return None
    content = msg.get("content")
    if isinstance(content, list):  # some gateways return content parts
        content = "".join(
            p.get("text", "")
            for p in content
            if isinstance(p, dict) and isinstance(p.get("text"), str)
        )
    return content if isinstance(content, str) and content.strip() else None


class OpenAICompatClient:
    """Minimal OpenAI-compatible client: ``GET /models`` and JSON chat completions."""

    def __init__(
        self,
        base_url: str,
        api_key: SecretStr | str,
        model: str,
        timeout_s: float,
        transport: httpx.BaseTransport | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self._key = api_key.get_secret_value() if isinstance(api_key, SecretStr) else api_key
        self.model = model
        self.timeout_s = timeout_s
        self._transport = transport

    def __repr__(self) -> str:
        return (
            f"OpenAICompatClient(base_url={self.base_url!r}, model={self.model!r}, key={REDACTED})"
        )

    @property
    def secrets(self) -> list[str]:
        return [self._key] if self._key else []

    def _scrub(self, text: str) -> str:
        return redact(text, self.secrets)

    def _client(self, timeout_s: float) -> httpx.Client:
        t = max(timeout_s, 0.1)
        return httpx.Client(
            timeout=httpx.Timeout(t, connect=min(CONNECT_TIMEOUT_S, t)),
            transport=self._transport,
            headers={"Authorization": f"Bearer {self._key}", "Accept": "application/json"},
            follow_redirects=False,
        )

    def _transport_error(self, e: Exception) -> tuple[ErrorCode, str]:
        if isinstance(e, httpx.TimeoutException):
            return "timeout", f"request timed out ({type(e).__name__})"
        if isinstance(e, httpx.TransportError):
            return "unreachable", self._scrub(f"cannot reach gateway ({type(e).__name__}: {e})")
        return "bad_response", self._scrub(f"{type(e).__name__}: {e}")

    def list_models(self, timeout_s: float | None = None) -> ModelsOutcome:
        t0 = time.monotonic()
        try:
            with self._client(timeout_s or self.timeout_s) as c:
                r = c.get(f"{self.base_url}/models")
        except (httpx.HTTPError, ValueError) as e:
            code, msg = self._transport_error(e)
            return ModelsOutcome(None, None, int((time.monotonic() - t0) * 1000), code, msg)
        ms = int((time.monotonic() - t0) * 1000)
        if r.status_code >= 400:
            code, msg = _classify_http(r.status_code, self._scrub(r.text))
            return ModelsOutcome(None, r.status_code, ms, code, msg)
        try:
            data = r.json()
            items = data.get("data") if isinstance(data, dict) else data
            ids = [str(m["id"]) for m in items if isinstance(m, dict) and "id" in m]
        except (ValueError, TypeError, KeyError):
            return ModelsOutcome(None, r.status_code, ms, "bad_response", "GET /models: not JSON")
        return ModelsOutcome(ids, r.status_code, ms)

    def chat_json(
        self,
        build_messages: Callable[[StructuredMode], list[dict[str, Any]]],
        schema: dict[str, Any],
        *,
        schema_name: str,
        modes: list[StructuredMode],
        deadline: float,
        on_attempt: Callable[[], None] | None = None,
    ) -> ChatOutcome:
        """Try ``modes`` in order until one yields a JSON object (downgrade on 'unsupported')."""
        t0 = time.monotonic()
        cache_key = (self.base_url, self.model)
        cached = _MODE_CACHE.get(cache_key)
        if cached in modes:
            modes = modes[modes.index(cached) :]
        out = ChatOutcome(ok=False)
        strict = to_strict_schema(schema)

        for i, mode in enumerate(modes):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                out.error_code, out.message = "timeout", "time budget exhausted"
                break
            body: dict[str, Any] = {"model": self.model, "messages": build_messages(mode)}
            if mode == "json_schema":
                body["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": schema_name, "schema": strict, "strict": True},
                }
            elif mode == "json_object":
                body["response_format"] = {"type": "json_object"}
            attempt: dict[str, Any] = {"mode": mode}
            out.attempts.append(attempt)
            try:
                with self._client(remaining) as c:
                    r = c.post(f"{self.base_url}/chat/completions", json=body)
            except (httpx.HTTPError, ValueError) as e:
                out.error_code, out.message = self._transport_error(e)
                attempt.update(error_code=out.error_code, message=out.message)
                break
            finally:
                if on_attempt is not None:
                    on_attempt()
            attempt["http_status"] = r.status_code
            text = self._scrub(r.text)
            if r.status_code >= 400:
                if _is_unsupported_mode(r.status_code, text) and i + 1 < len(modes):
                    attempt["message"] = f"mode unsupported: {_excerpt(text)}"
                    continue
                out.error_code, out.message = _classify_http(r.status_code, text)
                attempt.update(error_code=out.error_code, message=out.message)
                break
            try:
                data = r.json()
            except ValueError:
                out.error_code, out.message = "bad_response", "response is not JSON"
                attempt.update(error_code=out.error_code)
                break
            out.response = data if isinstance(data, dict) else None
            out.model = (data.get("model") if isinstance(data, dict) else None) or self.model
            content = _message_content(data)
            if content is None:
                out.error_code, out.message = "bad_response", "no message content in response"
                attempt.update(error_code=out.error_code)
                break
            obj, path = extract_json_object(content)
            if obj is None:
                out.error_code = "bad_response"
                out.message = f"no JSON object in answer: {_excerpt(self._scrub(content))}"
                attempt.update(error_code=out.error_code, parse_path=path)
                break
            out.ok, out.mode, out.payload, out.parse_path = True, mode, _strip_nulls(obj), path
            out.error_code = out.message = None
            attempt["parse_path"] = path
            _MODE_CACHE[cache_key] = mode
            break

        out.latency_ms = int((time.monotonic() - t0) * 1000)
        return out


def client_from_settings(
    settings: Settings, transport: httpx.BaseTransport | None = None, timeout_s: float | None = None
) -> OpenAICompatClient | None:
    """Client if base URL, key and model are all configured, else None."""
    if not (settings.llm_base_url and settings.llm_api_key and settings.llm_model):
        return None
    if not settings.llm_api_key.get_secret_value():
        return None
    return OpenAICompatClient(
        settings.llm_base_url,
        settings.llm_api_key,
        settings.llm_model,
        timeout_s or settings.llm_timeout_s,
        transport=transport,
    )


def missing_settings(settings: Settings) -> list[str]:
    """Env var names that must be set for ``openai_compat``."""
    out = []
    if not settings.llm_base_url:
        out.append("MIMIC_LLM_BASE_URL")
    if not settings.llm_api_key or not settings.llm_api_key.get_secret_value():
        out.append("MIMIC_LLM_API_KEY")
    if not settings.llm_model:
        out.append("MIMIC_LLM_MODEL")
    return out


# ---- interpreter ---------------------------------------------------------------------------------


class OpenAICompatInterpreter:
    """Labels elements through an OpenAI-compatible gateway; failure -> fallback result."""

    provider: InterpreterProvider = "openai_compat"

    def __init__(
        self,
        settings: Settings,
        on_progress: ProgressCallback | None = None,
        transport: httpx.BaseTransport | None = None,
    ):
        self.settings = settings
        self.client = client_from_settings(settings, transport=transport)
        self.model = settings.llm_model
        self.timeout_s = settings.llm_timeout_s
        self.modes = modes_for(settings.llm_supports_json_schema)
        self.on_progress = on_progress

    def __repr__(self) -> str:
        return f"OpenAICompatInterpreter(model={self.model!r})"

    def interpret(self, inp: InterpretationInput) -> InterpretationResult:
        t0 = time.monotonic()
        keyframes_dir = (inp.job_dir / "keyframes").resolve()
        raw: dict[str, Any] = {"provider": self.provider, "model": self.model}
        secrets = self.client.secrets if self.client else []

        def fail(status: str, reason: str) -> InterpretationResult:
            ms = int((time.monotonic() - t0) * 1000)
            reason = redact(reason, secrets)
            raw.update(status=status, error=reason, duration_ms=ms)
            write_raw(inp.job_dir, raw, secrets)
            return build_fallback(
                inp,
                status=status,  # type: ignore[arg-type]
                provider=self.provider,
                model=self.model,
                duration_ms=ms,
                notes=[reason],
            )

        if self.client is None:
            return fail("error", "LLM gateway not configured")
        if not inp.elements:
            return fail("fallback", "no elements to label")
        images = select_images(inp, keyframes_dir) if keyframes_dir.is_dir() else []
        encoded: list[tuple[Path, str]] = []
        for p in images:
            side = ELEMENT_IMAGE_MAX_SIDE if p.name.startswith("el_") else CONTEXT_IMAGE_MAX_SIDE
            url = encode_image(p, side)
            if url is not None:
                encoded.append((p, url))
        if not encoded:
            return fail("fallback", "no keyframes available to label")

        schema = build_schema([e.id for e in inp.elements])
        shown = [p for p, _ in encoded]
        image_parts = [image_part(u) for _, u in encoded]

        def build_messages(mode: StructuredMode) -> list[dict[str, Any]]:
            text = build_prompt(
                inp, shown, attached=True, schema=None if mode == "json_schema" else schema
            )
            return [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [{"type": "text", "text": text}, *image_parts]},
            ]

        def tick() -> None:
            if self.on_progress is not None:
                try:
                    self.on_progress(min((time.monotonic() - t0) / self.timeout_s, 1.0))
                except Exception:  # noqa: BLE001 - progress must never break labeling
                    pass

        outcome = self.client.chat_json(
            build_messages,
            schema,
            schema_name="mimic_labels",
            modes=self.modes,
            deadline=t0 + self.timeout_s,
            on_attempt=tick,
        )
        raw.update(
            mode=outcome.mode,
            attempts=outcome.attempts,
            parse_path=outcome.parse_path,
            images=[p.name for p in shown],
            response=_truncate_response(outcome.response),
        )
        if not outcome.ok:
            status = "timeout" if outcome.error_code == "timeout" else "error"
            return fail(status, f"{outcome.error_code}: {outcome.message}")

        try:
            payload = InterpretationPayload.model_validate(outcome.payload)
        except ValidationError as e:
            return fail("error", f"payload failed validation ({e.error_count()} errors)")

        base = build_fallback(inp, status="fallback", provider=self.provider)
        ms = int((time.monotonic() - t0) * 1000)
        result, issues = sanitize_payload(
            payload, inp, base, provider=self.provider, model=outcome.model, duration_ms=ms
        )
        raw["issues"] = issues
        if result is None:
            return fail("error", "AI output had no valid element labels")
        raw.update(status="ok", duration_ms=ms)
        write_raw(inp.job_dir, raw, secrets)
        return result


def _truncate_response(resp: dict[str, Any] | None) -> dict[str, Any] | None:
    """Keep the response for debugging unless it is unreasonably large."""
    if resp is None:
        return None
    try:
        if len(json.dumps(resp, default=str)) <= RESPONSE_RAW_MAX:
            return resp
    except (TypeError, ValueError):
        return None
    return {"truncated": True, "model": resp.get("model"), "usage": resp.get("usage")}
