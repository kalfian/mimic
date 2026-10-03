"""Server-side sessions in SQLite (PLAN-auth A1, A2, §2, §4).

The browser holds an opaque token (``secrets.token_urlsafe(32)``) in the ``mimic_session``
cookie; only ``sha256(token)`` is stored. A session is valid while

* its user is active (defence in depth on top of revocation),
* ``expires_at`` (absolute, set at sign-in) is in the future, and
* ``last_seen_at`` is newer than the idle timeout.

Resolving deletes a row that fails any of these. ``last_seen_at`` is written only when it is at
least :data:`TOUCH_INTERVAL_S` old, so the job polling doesn't turn every read into a write.

Never log a token or its hash.
"""

from __future__ import annotations

import hashlib
import secrets
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

from fastapi import Response

from app.auth.deps import SESSION_COOKIE
from app.auth.models import USER_COLUMNS, SessionRecord, UserRecord, user_from_row
from app.config import Settings
from app.core import db

#: Minimum age of ``last_seen_at`` before a request refreshes it.
TOUCH_INTERVAL_S = 60
#: Tokens longer than this can't be ours (``token_urlsafe(32)`` is 43 chars); skip hashing.
_MAX_TOKEN_CHARS = 256

Clock = Callable[[], datetime]

_USER_SELECT = ", ".join(f"u.{c.strip()}" for c in USER_COLUMNS.split(","))


def utcnow() -> datetime:
    return datetime.now(UTC)


def _iso(dt: datetime) -> str:
    # Same fixed precision as the job store, so stored text sorts chronologically.
    return dt.astimezone(UTC).isoformat(timespec="microseconds")


def hash_token(token: str) -> str:
    """``sha256(token)`` as hex: the only form that is stored."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class SqliteSessionStore:
    """:class:`app.auth.models.SessionStore` on ``data/mimic.db`` (table ``sessions``)."""

    def __init__(
        self,
        db_path: Path,
        *,
        idle_seconds: int,
        absolute_seconds: int,
        clock: Clock = utcnow,
        timeout_s: float = db.DEFAULT_TIMEOUT_S,
    ) -> None:
        if idle_seconds <= 0 or absolute_seconds <= 0:
            raise ValueError("session lifetimes must be positive")
        self.db_path = Path(db_path)
        self.idle = timedelta(seconds=idle_seconds)
        self.absolute = timedelta(seconds=absolute_seconds)
        self._clock = clock
        self._timeout_s = timeout_s

    @classmethod
    def from_settings(cls, settings: Settings, *, clock: Clock = utcnow) -> SqliteSessionStore:
        return cls(
            settings.db_path,
            idle_seconds=settings.session_idle_seconds,
            absolute_seconds=settings.session_absolute_seconds,
            clock=clock,
        )

    def _now(self) -> datetime:
        return self._clock().astimezone(UTC)

    # ---- writes -------------------------------------------------------------------------------

    def create(self, user_id: str) -> str:
        token = secrets.token_urlsafe(32)
        now = self._now()
        with db.connection(self.db_path, timeout_s=self._timeout_s) as conn:
            conn.execute(
                "INSERT INTO sessions (token_hash, user_id, created_at, last_seen_at, expires_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (hash_token(token), user_id, _iso(now), _iso(now), _iso(now + self.absolute)),
            )
        return token

    def revoke(self, token: str) -> None:
        if not token or len(token) > _MAX_TOKEN_CHARS:
            return
        with db.connection(self.db_path, timeout_s=self._timeout_s) as conn:
            conn.execute("DELETE FROM sessions WHERE token_hash = ?", (hash_token(token),))

    def revoke_all(self, user_id: str) -> int:
        with db.connection(self.db_path, timeout_s=self._timeout_s) as conn:
            return conn.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,)).rowcount

    def purge_expired(self) -> int:
        """Delete every session past its absolute or idle expiry. Returns the count."""
        now = self._now()
        with db.connection(self.db_path, timeout_s=self._timeout_s) as conn:
            return conn.execute(
                "DELETE FROM sessions WHERE expires_at <= ? OR last_seen_at <= ?",
                (_iso(now), _iso(now - self.idle)),
            ).rowcount

    # ---- reads --------------------------------------------------------------------------------

    def resolve(self, token: str) -> tuple[SessionRecord, UserRecord] | None:
        if not token or len(token) > _MAX_TOKEN_CHARS:
            return None
        token_hash = hash_token(token)
        now = self._now()
        with db.connection(self.db_path, timeout_s=self._timeout_s) as conn:
            row = conn.execute(
                "SELECT s.token_hash AS s_token_hash, s.user_id AS s_user_id, "
                "s.created_at AS s_created_at, s.last_seen_at AS s_last_seen_at, "
                f"s.expires_at AS s_expires_at, {_USER_SELECT} "
                "FROM sessions s JOIN users u ON u.id = s.user_id WHERE s.token_hash = ?",
                (token_hash,),
            ).fetchone()
            if row is None:
                return None
            session = SessionRecord(
                token_hash=row["s_token_hash"],
                user_id=row["s_user_id"],
                created_at=datetime.fromisoformat(row["s_created_at"]),
                last_seen_at=datetime.fromisoformat(row["s_last_seen_at"]),
                expires_at=datetime.fromisoformat(row["s_expires_at"]),
            )
            user = user_from_row(row)
            expired = session.expires_at <= now or session.last_seen_at <= now - self.idle
            if expired or not user.is_active:
                conn.execute("DELETE FROM sessions WHERE token_hash = ?", (token_hash,))
                return None
            if now - session.last_seen_at >= timedelta(seconds=TOUCH_INTERVAL_S):
                conn.execute(
                    "UPDATE sessions SET last_seen_at = ? WHERE token_hash = ?",
                    (_iso(now), token_hash),
                )
                session = SessionRecord(
                    token_hash=session.token_hash,
                    user_id=session.user_id,
                    created_at=session.created_at,
                    last_seen_at=now,
                    expires_at=session.expires_at,
                )
        return session, user


# ---- cookie helpers (A2) -------------------------------------------------------------------


def set_session_cookie(response: Response, token: str, settings: Settings) -> None:
    """``mimic_session``: HttpOnly, SameSite=Lax, Path=/, host-only, Max-Age = absolute
    lifetime, ``Secure`` when ``MIMIC_COOKIE_SECURE=1``."""
    response.set_cookie(
        SESSION_COOKIE,
        token,
        max_age=settings.session_absolute_seconds,
        path="/",
        httponly=True,
        samesite="lax",
        secure=settings.cookie_secure,
    )


def clear_session_cookie(response: Response, settings: Settings) -> None:
    """Expire the cookie (same path / samesite / secure as when it was set)."""
    response.delete_cookie(
        SESSION_COOKIE,
        path="/",
        httponly=True,
        samesite="lax",
        secure=settings.cookie_secure,
    )
