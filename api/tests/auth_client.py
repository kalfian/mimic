"""Real-login helpers for the B1 API tests (no dependency overrides).

:func:`create_account` stores a user with a *usable* password through the real stores, so the
tests exercise ``/api/auth/login`` and the session cookie end to end. Passwords here are test
fixtures, not credentials.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from app.auth.models import Role, UserRecord
from app.auth.passwords import hash_password
from app.auth.users import SqliteUserStore
from app.config import Settings
from app.core import db
from app.main import create_app

ADMIN_PW = "admin-password-for-tests"
USER_PW = "user-password-for-tests"


def _never_called(*_: Any) -> Any:
    raise AssertionError("the pipeline is not used by auth tests")


def create_account(
    db_path: Path,
    username: str,
    password: str,
    role: Role = Role.USER,
    *,
    must_change: bool = False,
    active: bool = True,
) -> UserRecord:
    db.migrate(db_path)
    store = SqliteUserStore(db_path)
    user = store.create(username, hash_password(password), role, must_change_password=must_change)
    if not active:
        with db.connection(db_path) as conn:
            conn.execute("UPDATE users SET is_active = 0 WHERE id = ?", (user.id,))
        user = store.get(user.id)
        assert user is not None
    return user


def started_client(settings: Settings) -> TestClient:
    """TestClient with the lifespan entered; the caller exits it."""
    client = TestClient(create_app(settings, pipeline=_never_called, reinterpret=_never_called))
    client.__enter__()
    return client


def login(client: TestClient, username: str, password: str, **kw: Any) -> Any:
    return client.post("/api/auth/login", json={"username": username, "password": password}, **kw)


def error_code(resp: Any) -> str:
    return resp.json()["error"]["code"]
