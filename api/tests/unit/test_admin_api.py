"""``/api/admin/*`` user management with real sessions (PLAN-auth §3.3, U2, A5, A13)."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.api.schemas import JobOptions
from app.auth.models import Role, UserRecord
from app.config import Settings
from app.core import db
from app.core.jobstore import SqliteJobStore
from tests.auth_client import ADMIN_PW, USER_PW, create_account, error_code, login, started_client

NEW_PW = "a brand new passphrase"
ClientFactory = Callable[[], TestClient]


@pytest.fixture
def new_client(settings: Settings) -> Iterator[ClientFactory]:
    """Independent clients (own cookie jar) on the same data dir."""
    opened: list[TestClient] = []

    def factory() -> TestClient:
        c = started_client(settings)
        opened.append(c)
        return c

    yield factory
    for c in opened:
        c.__exit__(None, None, None)


@pytest.fixture
def admin(settings: Settings) -> UserRecord:
    return create_account(settings.db_path, "root", ADMIN_PW, Role.ADMIN)


@pytest.fixture
def alice(settings: Settings, admin: UserRecord) -> UserRecord:
    return create_account(settings.db_path, "alice", USER_PW)


@pytest.fixture
def admin_client(new_client: ClientFactory, admin: UserRecord) -> TestClient:
    c = new_client()
    assert login(c, "root", ADMIN_PW).status_code == 200
    return c


def _signed_in(new_client: ClientFactory, username: str, password: str) -> TestClient:
    c = new_client()
    r = login(c, username, password)
    assert r.status_code == 200, r.text
    return c


def _sessions(db_path: Path, user_id: str) -> int:
    with db.connection(db_path) as conn:
        return conn.execute(
            "SELECT count(*) FROM sessions WHERE user_id = ?", (user_id,)
        ).fetchone()[0]


def _patch(c: TestClient, user_id: str, **body: Any) -> Any:
    return c.patch(f"/api/admin/users/{user_id}", json=body)


# ---- authorization -------------------------------------------------------------------------


def test_access_rules(new_client: ClientFactory, settings: Settings, alice: UserRecord) -> None:
    anon = new_client()
    r = anon.get("/api/admin/users")
    assert r.status_code == 401 and r.headers["cache-control"] == "no-store"
    user = _signed_in(new_client, "alice", USER_PW)
    r = user.get("/api/admin/users")
    assert r.status_code == 403 and error_code(r) == "forbidden"
    assert error_code(user.post("/api/admin/users", json={"username": "x" * 5})) == "forbidden"
    assert error_code(user.delete(f"/api/admin/users/{alice.id}")) == "forbidden"
    create_account(settings.db_path, "root2", ADMIN_PW, Role.ADMIN, must_change=True)
    forced = _signed_in(new_client, "root2", ADMIN_PW)
    assert error_code(forced.get("/api/admin/users")) == "password_change_required"


def test_list_users(admin_client: TestClient, alice: UserRecord, settings: Settings) -> None:
    SqliteJobStore(settings.db_path).create(
        "a" * 32, original_filename="a.mp4", ext="mp4", options=JobOptions(), owner_id=alice.id
    )
    r = admin_client.get("/api/admin/users")
    assert r.status_code == 200 and r.headers["cache-control"] == "no-store"
    items = r.json()["items"]
    assert [(i["username"], i["role"], i["job_count"]) for i in items] == [
        ("alice", "user", 1),
        ("root", "admin", 0),
    ]
    assert items[1]["last_login_at"] is not None
    assert "password_hash" not in items[0]


def test_unknown_or_malformed_user_id(admin_client: TestClient) -> None:
    for uid in ("0" * 32, "not-an-id", "A" * 32):
        assert admin_client.patch(f"/api/admin/users/{uid}", json={"role": "user"}).status_code == (
            404
        )
        assert admin_client.delete(f"/api/admin/users/{uid}").status_code == 404
        r = admin_client.post(f"/api/admin/users/{uid}/reset-password")
        assert r.status_code == 404 and error_code(r) == "not_found"


# ---- create --------------------------------------------------------------------------------


def test_create_user_temp_password_forces_change(
    admin_client: TestClient, new_client: ClientFactory, admin: UserRecord
) -> None:
    r = admin_client.post("/api/admin/users", json={"username": " Bob.Smith "})
    assert r.status_code == 201, r.text
    assert r.headers["cache-control"] == "no-store"
    body = r.json()
    temp = body["temporary_password"]
    assert len(temp) == 16
    user = body["user"]
    assert (user["username"], user["role"], user["is_active"], user["must_change_password"]) == (
        "bob.smith",
        "user",
        True,
        True,
    )
    assert user["job_count"] == 0

    bob = new_client()
    assert login(bob, "bob.smith", temp).json()["must_change_password"] is True
    assert error_code(bob.get("/api/admin/users")) == "password_change_required"
    r = bob.post("/api/auth/password", json={"current_password": temp, "new_password": NEW_PW})
    assert r.status_code == 200
    bob.post("/api/auth/logout")
    assert login(bob, "bob.smith", temp).status_code == 401  # temp works only until changed
    assert login(bob, "bob.smith", NEW_PW).status_code == 200


def test_create_user_errors(admin_client: TestClient, alice: UserRecord) -> None:
    r = admin_client.post("/api/admin/users", json={"username": "ALICE"})
    assert r.status_code == 409 and error_code(r) == "username_taken"
    for body in ({"username": "ab"}, {"username": "bad name"}, {"username": "carol", "role": "x"}):
        r = admin_client.post("/api/admin/users", json=body)
        assert r.status_code == 422 and error_code(r) == "invalid_request"
    r = admin_client.post("/api/admin/users", json={"username": "carol", "role": "admin"})
    assert r.status_code == 201 and r.json()["user"]["role"] == "admin"


# ---- update --------------------------------------------------------------------------------


def test_disable_revokes_immediately(
    admin_client: TestClient, new_client: ClientFactory, alice: UserRecord
) -> None:
    user = _signed_in(new_client, "alice", USER_PW)
    assert user.get("/api/auth/me").status_code == 200
    r = _patch(admin_client, alice.id, is_active=False)
    assert r.status_code == 200 and r.json()["is_active"] is False
    assert error_code(user.get("/api/auth/me")) == "unauthenticated"
    assert error_code(login(user, "alice", USER_PW)) == "account_disabled"
    # enable: no session is created, the user signs in again
    r = _patch(admin_client, alice.id, is_active=True)
    assert r.status_code == 200 and r.json()["is_active"] is True
    assert user.get("/api/auth/me").status_code == 401
    assert login(user, "alice", USER_PW).status_code == 200


def test_role_change_revokes_sessions(
    admin_client: TestClient, new_client: ClientFactory, alice: UserRecord, settings: Settings
) -> None:
    user = _signed_in(new_client, "alice", USER_PW)
    r = _patch(admin_client, alice.id, role="admin")
    assert r.status_code == 200 and r.json()["role"] == "admin"
    assert user.get("/api/auth/me").status_code == 401
    assert _sessions(settings.db_path, alice.id) == 0
    login(user, "alice", USER_PW)
    assert user.get("/api/admin/users").status_code == 200


def test_noop_update_keeps_sessions(
    admin_client: TestClient, new_client: ClientFactory, alice: UserRecord
) -> None:
    user = _signed_in(new_client, "alice", USER_PW)
    items = admin_client.get("/api/admin/users").json()["items"]
    before = next(i for i in items if i["id"] == alice.id)
    r = _patch(admin_client, alice.id, role="user", is_active=True)
    assert r.status_code == 200 and r.json() == before  # unchanged, updated_at not bumped
    assert user.get("/api/auth/me").status_code == 200


def test_update_body_validation(admin_client: TestClient, alice: UserRecord) -> None:
    for body in ({}, {"role": None, "is_active": None}, {"role": "root"}, {"extra": 1}):
        r = _patch(admin_client, alice.id, **body)
        assert r.status_code == 422 and error_code(r) == "invalid_request", body


def test_self_guards(admin_client: TestClient, admin: UserRecord) -> None:
    r = _patch(admin_client, admin.id, is_active=False)
    assert r.status_code == 409 and error_code(r) == "self_action_forbidden"
    r = admin_client.post(f"/api/admin/users/{admin.id}/reset-password")
    assert r.status_code == 409 and error_code(r) == "self_action_forbidden"
    r = admin_client.delete(f"/api/admin/users/{admin.id}")
    assert r.status_code == 409 and error_code(r) == "self_action_forbidden"
    # own role change is allowed in principle, but this is the last active admin
    r = _patch(admin_client, admin.id, role="user")
    assert r.status_code == 409 and error_code(r) == "last_admin"


def test_last_admin_guards(
    admin_client: TestClient, new_client: ClientFactory, settings: Settings, admin: UserRecord
) -> None:
    other = create_account(settings.db_path, "root2", ADMIN_PW, Role.ADMIN)
    other_client = _signed_in(new_client, "root2", ADMIN_PW)
    # root2 demotes root: fine (root2 remains)
    r = _patch(other_client, admin.id, role="user")
    assert r.status_code == 200
    assert admin_client.get("/api/auth/me").status_code == 401  # sessions revoked
    # now root2 is the last active admin: nobody can demote/disable/delete it
    assert error_code(_patch(other_client, other.id, role="user")) == "last_admin"
    # the refused self-demotion left root2's session intact
    assert other_client.get("/api/admin/users").status_code == 200


def test_last_admin_delete_by_other_admin(
    new_client: ClientFactory, settings: Settings, admin: UserRecord
) -> None:
    # root2 may delete root (another active admin remains); root2 can't delete itself
    create_account(settings.db_path, "root2", ADMIN_PW, Role.ADMIN)
    c2 = _signed_in(new_client, "root2", ADMIN_PW)
    assert c2.delete(f"/api/admin/users/{admin.id}").status_code == 204
    me = c2.get("/api/auth/me").json()
    assert error_code(c2.delete(f"/api/admin/users/{me['id']}")) == "self_action_forbidden"


# ---- reset password ------------------------------------------------------------------------


def test_reset_password(
    admin_client: TestClient, new_client: ClientFactory, alice: UserRecord
) -> None:
    user = _signed_in(new_client, "alice", USER_PW)
    r = admin_client.post(f"/api/admin/users/{alice.id}/reset-password")
    assert r.status_code == 200 and r.headers["cache-control"] == "no-store"
    temp = r.json()["temporary_password"]
    assert r.json()["user"]["must_change_password"] is True
    assert user.get("/api/auth/me").status_code == 401  # sessions revoked
    assert login(user, "alice", USER_PW).status_code == 401
    assert login(user, "alice", temp).json()["must_change_password"] is True


# ---- delete --------------------------------------------------------------------------------


def test_delete_user_removes_rows_dirs_sessions(
    admin_client: TestClient, new_client: ClientFactory, alice: UserRecord, settings: Settings
) -> None:
    store = SqliteJobStore(settings.db_path)
    kept_id, gone_ids = "c" * 32, ["a" * 32, "b" * 32]
    for job_id, owner in ((gone_ids[0], alice.id), (gone_ids[1], alice.id), (kept_id, None)):
        store.create(
            job_id, original_filename="a.mp4", ext="mp4", options=JobOptions(), owner_id=owner
        )
        (settings.jobs_dir / job_id).mkdir(parents=True)
        (settings.jobs_dir / job_id / "input.mp4").write_bytes(b"x")
    user = _signed_in(new_client, "alice", USER_PW)

    r = admin_client.delete(f"/api/admin/users/{alice.id}")
    assert r.status_code == 204 and r.content == b""
    assert r.headers["cache-control"] == "no-store"
    assert all(store.get(j) is None and not (settings.jobs_dir / j).exists() for j in gone_ids)
    assert store.get(kept_id) is not None and (settings.jobs_dir / kept_id).is_dir()
    assert _sessions(settings.db_path, alice.id) == 0
    assert user.get("/api/auth/me").status_code == 401
    assert login(user, "alice", USER_PW).status_code == 401
    names = [i["username"] for i in admin_client.get("/api/admin/users").json()["items"]]
    assert names == ["root"]


def test_delete_survives_missing_job_dir(
    admin_client: TestClient, alice: UserRecord, settings: Settings
) -> None:
    SqliteJobStore(settings.db_path).create(
        "a" * 32, original_filename="a.mp4", ext="mp4", options=JobOptions(), owner_id=alice.id
    )
    assert admin_client.delete(f"/api/admin/users/{alice.id}").status_code == 204


def test_admin_origin_check(admin_client: TestClient, alice: UserRecord) -> None:
    r = admin_client.delete(
        f"/api/admin/users/{alice.id}", headers={"Origin": "http://evil.localhost:5555"}
    )
    assert r.status_code == 403 and error_code(r) == "origin_not_allowed"
    assert admin_client.get("/api/admin/users").json()["items"][0]["username"] == "alice"


def test_temp_passwords_not_logged(
    admin_client: TestClient, alice: UserRecord, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level("DEBUG")
    t1 = admin_client.post("/api/admin/users", json={"username": "dave"}).json()
    t2 = admin_client.post(f"/api/admin/users/{alice.id}/reset-password").json()
    for secret in (t1["temporary_password"], t2["temporary_password"]):
        assert secret not in caplog.text
    assert f"reset the password of user {alice.id}" in caplog.text
