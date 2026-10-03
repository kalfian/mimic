"""SQLite session store (PLAN-auth A1, §2, §4)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from app.auth.models import Role
from app.auth.sessions import TOUCH_INTERVAL_S, SqliteSessionStore, hash_token
from app.core import db
from tests.auth_helpers import insert_user

IDLE_S = 3600
ABSOLUTE_S = 4 * 3600


class Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 10, 3, 12, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    path = tmp_path / "mimic.db"
    db.migrate(path)
    return path


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def store(db_path: Path, clock: Clock) -> SqliteSessionStore:
    return SqliteSessionStore(
        db_path, idle_seconds=IDLE_S, absolute_seconds=ABSOLUTE_S, clock=clock
    )


def _rows(db_path: Path) -> list[dict[str, str]]:
    with db.connection(db_path) as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM sessions")]


def test_create_and_resolve(db_path: Path, store: SqliteSessionStore) -> None:
    uid = insert_user(db_path, "alice")
    token = store.create(uid)
    resolved = store.resolve(token)
    assert resolved is not None
    session, user = resolved
    assert user.id == uid and user.username == "alice" and session.user_id == uid
    assert session.expires_at - session.created_at == timedelta(seconds=ABSOLUTE_S)


def test_only_the_hash_is_stored(db_path: Path, store: SqliteSessionStore) -> None:
    token = store.create(insert_user(db_path, "alice"))
    rows = _rows(db_path)
    assert len(rows) == 1 and rows[0]["token_hash"] == hash_token(token)
    assert all(token not in str(v) for v in rows[0].values())
    assert token not in repr(store.resolve(token))


def test_unknown_and_garbage_tokens(db_path: Path, store: SqliteSessionStore) -> None:
    store.create(insert_user(db_path, "alice"))
    assert store.resolve("nope") is None
    assert store.resolve("") is None
    assert store.resolve("x" * 10_000) is None


def test_idle_expiry_deletes_row(db_path: Path, store: SqliteSessionStore, clock: Clock) -> None:
    token = store.create(insert_user(db_path, "alice"))
    clock.advance(IDLE_S - 1)
    assert store.resolve(token) is not None  # touches last_seen
    clock.advance(IDLE_S - 1)
    assert store.resolve(token) is not None
    clock.advance(IDLE_S)
    assert store.resolve(token) is None
    assert _rows(db_path) == []


def test_absolute_expiry(db_path: Path, store: SqliteSessionStore, clock: Clock) -> None:
    token = store.create(insert_user(db_path, "alice"))
    for _ in range(4):  # keep it active, but the absolute limit still applies
        clock.advance(IDLE_S - 60)
        assert store.resolve(token) is not None
    clock.advance(4 * 60 + 1)
    assert store.resolve(token) is None
    assert _rows(db_path) == []


def test_last_seen_throttle(db_path: Path, store: SqliteSessionStore, clock: Clock) -> None:
    token = store.create(insert_user(db_path, "alice"))
    first = _rows(db_path)[0]["last_seen_at"]
    clock.advance(TOUCH_INTERVAL_S - 1)
    store.resolve(token)
    assert _rows(db_path)[0]["last_seen_at"] == first
    clock.advance(1)
    resolved = store.resolve(token)
    assert resolved is not None and resolved[0].last_seen_at == clock.now
    assert _rows(db_path)[0]["last_seen_at"] != first


def test_revoke_and_revoke_all(db_path: Path, store: SqliteSessionStore) -> None:
    alice, bob = insert_user(db_path, "alice"), insert_user(db_path, "bob")
    a1, a2, b1 = store.create(alice), store.create(alice), store.create(bob)
    store.revoke(a1)
    assert store.resolve(a1) is None and store.resolve(a2) is not None
    store.revoke("unknown-token")  # no error
    assert store.revoke_all(alice) == 1
    assert store.resolve(a2) is None and store.resolve(b1) is not None


def test_disabled_user_never_resolves(db_path: Path, store: SqliteSessionStore) -> None:
    uid = insert_user(db_path, "alice")
    token = store.create(uid)
    with db.connection(db_path) as conn:
        conn.execute("UPDATE users SET is_active = 0 WHERE id = ?", (uid,))
    assert store.resolve(token) is None
    assert _rows(db_path) == []


def test_role_comes_from_users_row(db_path: Path, store: SqliteSessionStore) -> None:
    uid = insert_user(db_path, "alice")
    token = store.create(uid)
    with db.connection(db_path) as conn:
        conn.execute("UPDATE users SET role = 'admin' WHERE id = ?", (uid,))
    resolved = store.resolve(token)
    assert resolved is not None and resolved[1].role is Role.ADMIN


def test_cascade_on_user_delete(db_path: Path, store: SqliteSessionStore) -> None:
    uid = insert_user(db_path, "alice")
    store.create(uid)
    with db.connection(db_path) as conn:
        conn.execute("DELETE FROM users WHERE id = ?", (uid,))
    assert _rows(db_path) == []


def test_purge_expired(db_path: Path, store: SqliteSessionStore, clock: Clock) -> None:
    uid = insert_user(db_path, "alice")
    old = store.create(uid)
    clock.advance(IDLE_S - 10)
    fresh = store.create(uid)
    clock.advance(20)  # `old` is now idle-expired, `fresh` is not
    assert store.purge_expired() == 1
    assert store.resolve(fresh) is not None
    assert store.resolve(old) is None


def test_invalid_lifetimes(db_path: Path) -> None:
    with pytest.raises(ValueError):
        SqliteSessionStore(db_path, idle_seconds=0, absolute_seconds=10)
