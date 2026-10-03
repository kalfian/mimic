"""``/api/auth/*`` end to end with real sessions (PLAN-auth §3.3, §5, A2, A6, A7, A8, A11)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.auth.deps import SESSION_COOKIE
from app.auth.models import Role, UserRecord
from app.auth.users import SqliteUserStore
from app.config import Settings, get_settings
from app.core import db
from tests.auth_client import (
    ADMIN_PW,
    USER_PW,
    create_account,
    error_code,
    login,
    started_client,
)

NEW_PW = "a brand new passphrase"


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    c = started_client(settings)
    yield c
    c.__exit__(None, None, None)


@pytest.fixture
def db_path(settings: Settings) -> Path:
    return settings.db_path


@pytest.fixture
def admin(db_path: Path) -> UserRecord:
    return create_account(db_path, "root", ADMIN_PW, Role.ADMIN)


@pytest.fixture
def alice(db_path: Path, admin: UserRecord) -> UserRecord:
    return create_account(db_path, "alice", USER_PW)


def _session_rows(db_path: Path) -> int:
    with db.connection(db_path) as conn:
        return conn.execute("SELECT count(*) FROM sessions").fetchone()[0]


# ---- setup / status ------------------------------------------------------------------------


def test_status_and_setup_required(client: TestClient, db_path: Path) -> None:
    r = client.get("/api/auth/status")
    assert r.status_code == 200 and r.json() == {"setup_required": True}
    assert r.headers["cache-control"] == "no-store"
    r = login(client, "root", ADMIN_PW)
    assert r.status_code == 409 and error_code(r) == "setup_required"
    assert "make create-admin" in r.json()["error"]["message"]
    assert r.headers["cache-control"] == "no-store"
    create_account(db_path, "root", ADMIN_PW, Role.ADMIN)
    assert client.get("/api/auth/status").json() == {"setup_required": False}


# ---- login lifecycle -----------------------------------------------------------------------


def test_login_me_logout(client: TestClient, alice: UserRecord) -> None:
    r = login(client, "  Alice ", USER_PW)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["id"] == alice.id and body["username"] == "alice" and body["role"] == "user"
    assert body["must_change_password"] is False
    assert r.headers["cache-control"] == "no-store"

    me = client.get("/api/auth/me")
    assert me.status_code == 200 and me.json() == body
    assert me.headers["cache-control"] == "no-store"

    out = client.post("/api/auth/logout")
    assert out.status_code == 204 and out.content == b""
    assert "max-age=0" in out.headers["set-cookie"].lower()
    assert client.get("/api/auth/me").status_code == 401
    # idempotent without a session
    assert client.post("/api/auth/logout").status_code == 204


def test_logout_revokes_server_side(client: TestClient, alice: UserRecord, db_path: Path) -> None:
    login(client, "alice", USER_PW)
    token = client.cookies.get(SESSION_COOKIE)
    assert token and _session_rows(db_path) == 1
    client.post("/api/auth/logout")
    assert _session_rows(db_path) == 0
    client.cookies.set(SESSION_COOKIE, token)  # replaying the old cookie fails
    assert error_code(client.get("/api/auth/me")) == "unauthenticated"


def test_cookie_attributes(client: TestClient, alice: UserRecord, settings: Settings) -> None:
    r = login(client, "alice", USER_PW)
    cookie = r.headers["set-cookie"]
    parts = [p.strip().lower() for p in cookie.split(";")]
    assert parts[0].startswith(f"{SESSION_COOKIE}=")
    assert "httponly" in parts and "samesite=lax" in parts and "path=/" in parts
    assert f"max-age={settings.session_absolute_seconds}" in parts
    assert "secure" not in parts and not any(p.startswith("domain=") for p in parts)


def test_secure_cookie_setting(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, db_path: Path
) -> None:
    monkeypatch.setenv("MIMIC_COOKIE_SECURE", "1")
    get_settings.cache_clear()
    secure = get_settings()
    create_account(db_path, "alice", USER_PW, Role.ADMIN)
    c = started_client(secure)
    try:
        r = login(c, "alice", USER_PW)
        assert "secure" in [p.strip().lower() for p in r.headers["set-cookie"].split(";")]
    finally:
        c.__exit__(None, None, None)


def test_invalid_credentials(client: TestClient, alice: UserRecord) -> None:
    for name, pw in (("alice", "wrong-password-123"), ("nobody", USER_PW)):
        r = login(client, name, pw)
        assert r.status_code == 401 and error_code(r) == "invalid_credentials"
        assert "set-cookie" not in r.headers


def test_invalid_login_body(client: TestClient, alice: UserRecord) -> None:
    r = client.post("/api/auth/login", json={"username": "", "password": "x"})
    assert r.status_code == 422 and error_code(r) == "invalid_request"
    r = client.post("/api/auth/login", json={"username": "a", "password": "x", "extra": 1})
    assert r.status_code == 422


def test_account_disabled_only_after_correct_password(client: TestClient, db_path: Path) -> None:
    create_account(db_path, "root", ADMIN_PW, Role.ADMIN)
    create_account(db_path, "gone", USER_PW, active=False)
    assert error_code(login(client, "gone", "wrong-password-123")) == "invalid_credentials"
    r = login(client, "gone", USER_PW)
    assert r.status_code == 403 and error_code(r) == "account_disabled"


def test_login_replaces_presented_session(
    client: TestClient, alice: UserRecord, db_path: Path
) -> None:
    login(client, "alice", USER_PW)
    first = client.cookies.get(SESSION_COOKIE)
    login(client, "alice", USER_PW)
    assert client.cookies.get(SESSION_COOKIE) != first
    assert _session_rows(db_path) == 1


def test_login_rehashes_outdated_hash(
    client: TestClient, alice: UserRecord, db_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.auth import policy

    before = SqliteUserStore(db_path).get(alice.id)
    assert before is not None and before.password_hash.startswith("scrypt$10$")
    monkeypatch.setattr(policy, "SCRYPT_PARAMS", policy.ScryptParams(log2_n=11))
    assert login(client, "alice", USER_PW).status_code == 200
    after = SqliteUserStore(db_path).get(alice.id)
    assert after is not None and after.password_hash.startswith("scrypt$11$")
    assert after.last_login_at is not None


def test_rate_limit_and_retry_after(client: TestClient, alice: UserRecord) -> None:
    for _ in range(5):
        assert login(client, "alice", "wrong-password-123").status_code == 401
    r = login(client, "alice", USER_PW)  # even the right password is throttled now
    assert r.status_code == 429 and error_code(r) == "too_many_attempts"
    assert 1 <= int(r.headers["retry-after"]) <= 900
    # a different username from the same client is not blocked
    assert login(client, "root", ADMIN_PW).status_code == 200


def test_success_resets_key(client: TestClient, alice: UserRecord) -> None:
    for _ in range(4):
        login(client, "alice", "wrong-password-123")
    assert login(client, "alice", USER_PW).status_code == 200
    for _ in range(4):
        login(client, "alice", "wrong-password-123")
    assert login(client, "alice", USER_PW).status_code == 200


# ---- forced change + password change -------------------------------------------------------


def test_forced_change_flow(client: TestClient, db_path: Path) -> None:
    create_account(db_path, "root", ADMIN_PW, Role.ADMIN)
    create_account(db_path, "newbie", USER_PW, must_change=True)
    r = login(client, "newbie", USER_PW)
    assert r.status_code == 200 and r.json()["must_change_password"] is True
    assert client.get("/api/auth/me").status_code == 200  # allowed while forced
    assert error_code(client.get("/api/admin/users")) == "password_change_required"

    old = client.cookies.get(SESSION_COOKIE)
    r = client.post(
        "/api/auth/password", json={"current_password": USER_PW, "new_password": NEW_PW}
    )
    assert r.status_code == 200, r.text
    assert r.json()["must_change_password"] is False
    assert r.headers["cache-control"] == "no-store"
    new = client.cookies.get(SESSION_COOKIE)
    assert new and new != old
    assert client.get("/api/auth/me").json()["must_change_password"] is False
    assert _session_rows(db_path) == 1

    client.post("/api/auth/logout")
    assert login(client, "newbie", USER_PW).status_code == 401
    assert login(client, "newbie", NEW_PW).status_code == 200


def test_password_change_revokes_other_sessions(
    settings: Settings, client: TestClient, alice: UserRecord
) -> None:
    other = started_client(settings)
    try:
        login(other, "alice", USER_PW)
        login(client, "alice", USER_PW)
        r = client.post(
            "/api/auth/password", json={"current_password": USER_PW, "new_password": NEW_PW}
        )
        assert r.status_code == 200
        assert other.get("/api/auth/me").status_code == 401
        assert client.get("/api/auth/me").status_code == 200
    finally:
        other.__exit__(None, None, None)


def test_password_change_errors(client: TestClient, alice: UserRecord) -> None:
    assert (
        client.post(
            "/api/auth/password", json={"current_password": USER_PW, "new_password": NEW_PW}
        ).status_code
        == 401
    )
    login(client, "alice", USER_PW)

    def change(current: str, new: str) -> Any:
        return client.post(
            "/api/auth/password", json={"current_password": current, "new_password": new}
        )

    r = change("wrong-password-123", NEW_PW)
    assert r.status_code == 422 and error_code(r) == "current_password_incorrect"
    for new, needle in (("short", "at least 12"), ("Alice", "at least"), (USER_PW, "different")):
        r = change(USER_PW, new)
        assert r.status_code == 422 and error_code(r) == "weak_password"
        assert needle in r.json()["error"]["message"]
    # still signed in, password unchanged
    assert client.get("/api/auth/me").status_code == 200


def test_password_change_rate_limited(client: TestClient, alice: UserRecord) -> None:
    login(client, "alice", USER_PW)
    for _ in range(5):
        client.post(
            "/api/auth/password",
            json={"current_password": "wrong-password-123", "new_password": NEW_PW},
        )
    r = client.post(
        "/api/auth/password", json={"current_password": USER_PW, "new_password": NEW_PW}
    )
    assert r.status_code == 429 and "retry-after" in r.headers


# ---- origin check (CSRF) + CORS ------------------------------------------------------------


@pytest.mark.parametrize(
    ("origin", "status"),
    [
        (None, 204),
        ("http://localhost:3000", 204),  # MIMIC_CORS_ORIGINS default
        ("HTTP://LOCALHOST:3000", 204),
        ("http://testserver", 204),  # the API's own origin (Swagger /docs)
        ("http://evil.localhost:5555", 403),
        ("http://localhost:5555", 403),  # same-site but not allowed
        ("null", 403),
        ("https://localhost:3000", 403),
    ],
)
def test_origin_check(client: TestClient, origin: str | None, status: int) -> None:
    headers = {"Origin": origin} if origin is not None else {}
    r = client.post("/api/auth/logout", headers=headers)
    assert r.status_code == status, r.text
    if status == 403:
        assert error_code(r) == "origin_not_allowed"


def test_origin_check_applies_to_login_and_skips_safe_methods(
    client: TestClient, alice: UserRecord
) -> None:
    evil = {"Origin": "http://evil.localhost:5555"}
    r = login(client, "alice", USER_PW, headers=evil)
    assert r.status_code == 403 and "set-cookie" not in r.headers
    assert client.get("/api/auth/status", headers=evil).status_code == 200


def test_cors_preflight_with_credentials(client: TestClient) -> None:
    r = client.options(
        "/api/auth/login",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "PATCH",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert r.status_code == 200
    assert r.headers["access-control-allow-origin"] == "http://localhost:3000"
    assert r.headers["access-control-allow-credentials"] == "true"
    methods = {m.strip() for m in r.headers["access-control-allow-methods"].split(",")}
    assert {"GET", "POST", "PATCH", "DELETE"} <= methods


def test_cors_rejection_body_readable_from_allowed_origin(client: TestClient) -> None:
    r = client.get("/api/auth/me", headers={"Origin": "http://localhost:3000"})
    assert r.status_code == 401
    assert r.headers["access-control-allow-origin"] == "http://localhost:3000"
    assert r.headers["access-control-allow-credentials"] == "true"


def test_wildcard_origin_refuses_startup(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MIMIC_CORS_ORIGINS", "*")
    get_settings.cache_clear()
    from app.main import create_app

    with pytest.raises(ValidationError):
        create_app()


def test_http_status_fallback_codes() -> None:
    from app.core.errors import ErrorCode
    from app.main import _HTTP_STATUS_CODES

    assert _HTTP_STATUS_CODES[401] is ErrorCode.UNAUTHENTICATED
    assert _HTTP_STATUS_CODES[403] is ErrorCode.FORBIDDEN
    assert _HTTP_STATUS_CODES[429] is ErrorCode.TOO_MANY_ATTEMPTS


def test_no_secrets_in_logs(
    client: TestClient, alice: UserRecord, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level("DEBUG")
    login(client, "alice", "wrong-password-123")
    login(client, "alice", USER_PW)
    token = client.cookies.get(SESSION_COOKIE)
    client.post("/api/auth/password", json={"current_password": USER_PW, "new_password": NEW_PW})
    text = caplog.text
    assert token
    for secret in (USER_PW, NEW_PW, "wrong-password-123", token):
        assert secret not in text
    assert "alice" not in text  # failed-attempt lines carry no username
    assert "failed sign-in" in text and "signed in" in text  # the audit lines were captured
