"""Test helpers for accounts (PLAN-auth §9).

* :func:`insert_user` writes a ``users`` row with raw SQL and an **unusable** password hash
  (nobody can sign in with it). Use the real stores / login flow when a password must work.
* :func:`as_user` makes the app's auth dependencies return a given user (or behave as
  anonymous) through ``app.dependency_overrides``, so route tests need no session cookie.
  ``get_current_user`` / ``require_admin`` still run their real checks on that user.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI

from app.auth.deps import get_optional_user, get_session_user
from app.auth.models import USER_COLUMNS, Role, UserRecord, user_from_row
from app.core import db
from app.core.errors import ErrorCode, PipelineError

#: Never a valid ``scrypt$...`` encoding, so verification always fails.
UNUSABLE_PASSWORD_HASH = "!unusable"


def insert_user(
    db_path: Path,
    username: str,
    role: Role | str = Role.USER,
    *,
    active: bool = True,
    must_change: bool = False,
    user_id: str | None = None,
) -> str:
    """Insert a user (migrating the database first if needed). Returns its id."""
    db.migrate(db_path)
    user_id = user_id or uuid.uuid4().hex
    now = datetime.now(UTC).isoformat(timespec="microseconds")
    with db.connection(db_path) as conn:
        conn.execute(
            f"INSERT INTO users ({USER_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL)",
            (
                user_id,
                username,
                UNUSABLE_PASSWORD_HASH,
                Role(role).value,
                int(active),
                int(must_change),
                now,
                now,
                now,
            ),
        )
    return user_id


def fetch_user(db_path: Path, user_id: str) -> UserRecord:
    with db.connection(db_path) as conn:
        row = conn.execute(f"SELECT {USER_COLUMNS} FROM users WHERE id = ?", (user_id,)).fetchone()
    assert row is not None, f"no user {user_id}"
    return user_from_row(row)


def make_user(db_path: Path, username: str, role: Role | str = Role.USER, **kw: Any) -> UserRecord:
    """:func:`insert_user` + :func:`fetch_user`."""
    return fetch_user(db_path, insert_user(db_path, username, role, **kw))


@contextmanager
def as_user(app: FastAPI, user: UserRecord | None) -> Iterator[None]:
    """Within the block, requests to ``app`` are authenticated as ``user``.

    ``None`` = anonymous (401 on protected routes, ``OptionalUserDep`` is None). A disabled
    user behaves like a revoked session (401). A user with ``must_change_password`` gets a
    session but no full access (403 ``password_change_required`` from ``CurrentUserDep``).
    Previous overrides are restored on exit.
    """

    def session_user() -> UserRecord:
        if user is None or not user.is_active:
            raise PipelineError(ErrorCode.UNAUTHENTICATED)
        return user

    def optional_user() -> UserRecord | None:
        return user if user is not None and user.has_full_access else None

    overrides = app.dependency_overrides
    saved = {dep: overrides.get(dep) for dep in (get_session_user, get_optional_user)}
    overrides[get_session_user] = session_user
    overrides[get_optional_user] = optional_user
    try:
        yield
    finally:
        for dep, previous in saved.items():
            if previous is None:
                overrides.pop(dep, None)
            else:
                overrides[dep] = previous
