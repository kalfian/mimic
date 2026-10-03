"""Auth domain records, store interfaces and the per-app service bundle (PLAN-auth §7).

``AuthServices`` is what ``app.main``'s lifespan stores on ``app.state.auth`` and what
``app.auth.deps.get_auth`` returns. The store Protocols are the interfaces the request
dependencies and routes are written against; ``app/auth/{users,sessions,ratelimit}.py`` (B1)
implement them.

Secrets: ``UserRecord.password_hash`` and ``SessionRecord.token_hash`` are excluded from
``repr`` so they never end up in logs or assertion output.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from app.config import Settings


class Role(StrEnum):
    """Account role. Same values as the contract literal ``UserRole`` (``api/schemas.py``)."""

    ADMIN = "admin"
    USER = "user"


#: Column list for ``SELECT`` into :func:`user_from_row` (table ``users``, ``core/db.py``).
USER_COLUMNS = (
    "id, username, password_hash, role, is_active, must_change_password, created_at, "
    "updated_at, password_changed_at, last_login_at, created_by"
)


@dataclass(frozen=True, slots=True)
class UserRecord:
    """One row of ``users``. Role and flags are always read fresh per request (§4)."""

    id: str
    username: str
    role: Role
    is_active: bool
    must_change_password: bool
    created_at: datetime
    updated_at: datetime
    password_changed_at: datetime
    last_login_at: datetime | None
    created_by: str | None
    password_hash: str = field(repr=False, compare=False)

    @property
    def is_admin(self) -> bool:
        return self.role is Role.ADMIN

    @property
    def has_full_access(self) -> bool:
        """Active and not in forced-password-change state ("full session" in §3.3)."""
        return self.is_active and not self.must_change_password


@dataclass(frozen=True, slots=True)
class SessionRecord:
    """One row of ``sessions``. The plaintext token is never stored or kept."""

    token_hash: str = field(repr=False)
    user_id: str
    created_at: datetime
    last_seen_at: datetime
    expires_at: datetime


def _dt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def user_from_row(row: sqlite3.Row) -> UserRecord:
    """Map a ``users`` row selected with :data:`USER_COLUMNS`."""
    created_at = _dt(row["created_at"])
    updated_at = _dt(row["updated_at"])
    password_changed_at = _dt(row["password_changed_at"])
    assert created_at and updated_at and password_changed_at
    return UserRecord(
        id=row["id"],
        username=row["username"],
        role=Role(row["role"]),
        is_active=bool(row["is_active"]),
        must_change_password=bool(row["must_change_password"]),
        created_at=created_at,
        updated_at=updated_at,
        password_changed_at=password_changed_at,
        last_login_at=_dt(row["last_login_at"]),
        created_by=row["created_by"],
        password_hash=row["password_hash"],
    )


# ---- store interfaces (implemented in B1) ----------------------------------------------------


@runtime_checkable
class UserStore(Protocol):
    """``users`` table. Guards raise ``PipelineError`` (``last_admin``, ``username_taken``,
    ``not_found``); every guard + write pair runs in one ``BEGIN IMMEDIATE`` transaction."""

    def create(
        self,
        username: str,
        password_hash: str,
        role: Role,
        *,
        must_change_password: bool,
        created_by: str | None = None,
    ) -> UserRecord: ...

    def get(self, user_id: str) -> UserRecord | None: ...

    def get_by_username(self, username: str) -> UserRecord | None: ...

    def list_with_job_counts(self) -> list[tuple[UserRecord, int]]:
        """Every user, sorted by username, with its number of jobs."""
        ...

    def update(
        self, user_id: str, *, role: Role | None = None, is_active: bool | None = None
    ) -> UserRecord:
        """Atomic last-active-admin guard on demote/disable."""
        ...

    def set_password(self, user_id: str, password_hash: str, *, must_change: bool) -> None: ...

    def record_login(self, user_id: str) -> None: ...

    def delete(self, user_id: str) -> list[str]:
        """Delete the user, its sessions and job rows in one transaction (atomic last-admin
        guard). Returns the removed job ids so the caller can delete their directories."""
        ...

    def has_active_admin(self) -> bool: ...


@runtime_checkable
class SessionStore(Protocol):
    """``sessions`` table. Only ``sha256(token)`` is stored."""

    def create(self, user_id: str) -> str:
        """New session; returns the plaintext token (shown once, put in the cookie)."""
        ...

    def resolve(self, token: str) -> tuple[SessionRecord, UserRecord] | None:
        """Valid session + its (active) user, or None. Deletes the row if it expired. Touches
        ``last_seen_at`` at most once a minute."""
        ...

    def revoke(self, token: str) -> None: ...

    def revoke_all(self, user_id: str) -> int: ...

    def purge_expired(self) -> int: ...


@runtime_checkable
class AttemptLimiter(Protocol):
    """In-memory sliding-window limiter for failed logins / password changes (A6).

    ``key`` is the normalized username (login) or ``"user:<id>"`` (password change).
    """

    def retry_after(self, ip: str, key: str) -> int | None:
        """Seconds until another attempt is allowed, or None if allowed now."""
        ...

    def record_failure(self, ip: str, key: str) -> None: ...

    def reset(self, ip: str, key: str) -> None:
        """Clear the ``(ip, key)`` window after a success."""
        ...


@dataclass(frozen=True, slots=True)
class AuthServices:
    """Per-app auth services (``app.state.auth``), built in ``app.main``'s lifespan."""

    users: UserStore
    sessions: SessionStore
    limiter: AttemptLimiter
    settings: Settings
