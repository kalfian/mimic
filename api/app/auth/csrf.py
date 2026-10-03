"""HTTP-level guards as pure ASGI middleware (PLAN-auth A7, §3).

* :class:`OriginCheckMiddleware` — CSRF defence: ``POST/PUT/PATCH/DELETE`` whose ``Origin``
  header is present but is neither an allowed CORS origin nor the API's own origin
  (``scheme://Host``, for Swagger ``/docs``) → 403 ``origin_not_allowed``. ``Origin: null`` is
  rejected. A request without ``Origin`` passes: browsers send it on every cross-origin request
  and every non-GET fetch/XHR/form POST, so its absence means a non-browser client.
  SameSite=Lax alone is not enough because every app on ``localhost:*`` is *same-site*.
* :class:`NoStoreMiddleware` — ``Cache-Control: no-store`` on every response (errors included)
  under the given path prefixes (``/api/auth/``, ``/api/admin/``).

Pure ASGI (not ``BaseHTTPMiddleware``) so the streaming upload body is never wrapped: both read
headers only. Register them *inside* ``CORSMiddleware`` (``add_middleware`` before CORS), so
preflights stay CORS's job and a rejection still carries CORS headers.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from urllib.parse import urlsplit

from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.core.errors import DEFAULT_MESSAGES, ERROR_HTTP_STATUS, ErrorCode

log = logging.getLogger(__name__)

UNSAFE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
_DEFAULT_PORTS = {"http": 80, "https": 443}


def normalize_origin(value: str) -> str | None:
    """Lowercase ``scheme://host[:port]`` with the default port dropped, or None if ``value``
    is not a plain http(s) origin (``null``, a path, credentials, garbage)."""
    try:
        parts = urlsplit(value.strip())
        port = parts.port
    except ValueError:
        return None
    scheme = parts.scheme.lower()
    host = parts.hostname
    if scheme not in _DEFAULT_PORTS or not host:
        return None
    if parts.path not in ("", "/") or parts.query or parts.fragment or parts.username:
        return None
    if ":" in host:  # IPv6 literal
        host = f"[{host}]"
    if port is None or port == _DEFAULT_PORTS[scheme]:
        return f"{scheme}://{host}"
    return f"{scheme}://{host}:{port}"


class OriginCheckMiddleware:
    """403 ``origin_not_allowed`` for unsafe requests from a foreign ``Origin``."""

    def __init__(self, app: ASGIApp, *, allowed_origins: Iterable[str]) -> None:
        self.app = app
        self.allowed = frozenset(o for o in map(normalize_origin, allowed_origins) if o)

    def _own_origin(self, scope: Scope, headers: Headers) -> str | None:
        host = headers.get("host")
        if not host:
            return None
        return normalize_origin(f"{scope.get('scheme', 'http')}://{host}")

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] not in UNSAFE_METHODS:
            await self.app(scope, receive, send)
            return
        headers = Headers(scope=scope)
        raw = headers.get("origin")
        if raw is None:
            await self.app(scope, receive, send)
            return
        origin = normalize_origin(raw)
        if origin is not None and (
            origin in self.allowed or origin == self._own_origin(scope, headers)
        ):
            await self.app(scope, receive, send)
            return
        log.warning(
            "origin check: rejected %s %s from a foreign origin",
            scope["method"],
            scope.get("path", ""),
        )
        await _send_error(send, ErrorCode.ORIGIN_NOT_ALLOWED)


async def _send_error(send: Send, code: ErrorCode) -> None:
    body = json.dumps({"error": {"code": code.value, "message": DEFAULT_MESSAGES[code]}}).encode()
    await send(
        {
            "type": "http.response.start",
            "status": ERROR_HTTP_STATUS[code],
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
                (b"cache-control", b"no-store"),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


class NoStoreMiddleware:
    """``Cache-Control: no-store`` on every response under ``prefixes``."""

    def __init__(self, app: ASGIApp, *, prefixes: Iterable[str]) -> None:
        self.app = app
        self.prefixes = tuple(prefixes)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not scope.get("path", "").startswith(self.prefixes):
            await self.app(scope, receive, send)
            return

        async def send_no_store(message: Message) -> None:
            if message["type"] == "http.response.start":
                message.setdefault("headers", [])
                MutableHeaders(scope=message)["Cache-Control"] = "no-store"
            await send(message)

        await self.app(scope, receive, send_no_store)
