"""In-memory login / password-change throttling (PLAN-auth A6).

Sliding window (default 15 min), counted on **failures** only:

* per ``(client IP, key)``: 5 failures → blocked until the oldest one leaves the window;
* per client IP (all keys): 30 failures → blocked likewise.

``key`` is the normalized username for logins and ``"user:<id>"`` for password changes. A
success clears the ``(ip, key)`` window (not the per-IP one). Keying on IP + username avoids
the "lock the admin out from anywhere" DoS of a username-only lockout. ``X-Forwarded-For`` is
not trusted (no proxy in this setup); callers pass ``request.client.host``.

Single-process and reset on restart, which is fine for a local tool (§12). Thread-safe: the
sync routes run on FastAPI's threadpool.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from collections.abc import Callable

DEFAULT_WINDOW_S = 15 * 60
DEFAULT_MAX_PER_KEY = 5
DEFAULT_MAX_PER_IP = 30
#: Sweep every window when this many keys are tracked, so the maps can't grow without bound.
_SWEEP_THRESHOLD = 10_000


class LoginLimiter:
    """:class:`app.auth.models.AttemptLimiter` with an injectable monotonic clock."""

    def __init__(
        self,
        *,
        window_s: float = DEFAULT_WINDOW_S,
        max_per_key: int = DEFAULT_MAX_PER_KEY,
        max_per_ip: int = DEFAULT_MAX_PER_IP,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if window_s <= 0 or max_per_key < 1 or max_per_ip < 1:
            raise ValueError("invalid limiter configuration")
        self.window_s = window_s
        self.max_per_key = max_per_key
        self.max_per_ip = max_per_ip
        self._clock = clock
        self._lock = threading.Lock()
        self._by_key: dict[tuple[str, str], deque[float]] = {}
        self._by_ip: dict[str, deque[float]] = {}

    # ---- internals (caller holds the lock) ----------------------------------------------------

    def _prune(self, q: deque[float], now: float) -> None:
        cutoff = now - self.window_s
        while q and q[0] <= cutoff:
            q.popleft()

    def _live(self, table: dict, key: object, now: float) -> deque[float] | None:
        q = table.get(key)
        if q is None:
            return None
        self._prune(q, now)
        if not q:
            del table[key]
            return None
        return q

    def _wait(self, q: deque[float] | None, limit: int, now: float) -> float:
        if q is None or len(q) < limit:
            return 0.0
        # Blocked until enough failures leave the window to drop below the limit.
        return q[len(q) - limit] + self.window_s - now

    def _sweep(self, now: float) -> None:
        for table in (self._by_key, self._by_ip):
            for key in list(table):
                self._live(table, key, now)

    # ---- AttemptLimiter -----------------------------------------------------------------------

    def retry_after(self, ip: str, key: str) -> int | None:
        now = self._clock()
        with self._lock:
            wait = max(
                self._wait(self._live(self._by_key, (ip, key), now), self.max_per_key, now),
                self._wait(self._live(self._by_ip, ip, now), self.max_per_ip, now),
            )
        if wait <= 0:
            return None
        return max(1, math.ceil(wait))

    def record_failure(self, ip: str, key: str) -> None:
        now = self._clock()
        with self._lock:
            if len(self._by_key) + len(self._by_ip) > _SWEEP_THRESHOLD:
                self._sweep(now)
            self._by_key.setdefault((ip, key), deque()).append(now)
            self._by_ip.setdefault(ip, deque()).append(now)

    def reset(self, ip: str, key: str) -> None:
        with self._lock:
            self._by_key.pop((ip, key), None)
