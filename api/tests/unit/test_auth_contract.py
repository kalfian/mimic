"""P0 auth contract: error codes, settings, policy, request/response models, dependency chain.

The dependency chain is exercised on a minimal app through ``tests/auth_helpers.as_user`` (the
same override mechanism B2's route tests use). Session resolution itself is B1's.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.api.schemas import (
    CreateUserRequest,
    Health,
    JobStatus,
    LoginRequest,
    UpdateUserRequest,
)
from app.auth import policy
from app.auth.deps import (
    SESSION_COOKIE,
    AdminDep,
    CurrentUserDep,
    OptionalUserDep,
    SessionUserDep,
)
from app.auth.models import AuthServices, Role, UserRecord
from app.config import Settings, get_settings
from app.core.errors import DEFAULT_MESSAGES, ERROR_HTTP_STATUS, ErrorCode, PipelineError
from tests.auth_helpers import UNUSABLE_PASSWORD_HASH, as_user, make_user

# ---- error codes -----------------------------------------------------------------------------

AUTH_CODES = {
    "unauthenticated": 401,
    "invalid_credentials": 401,
    "account_disabled": 403,
    "forbidden": 403,
    "password_change_required": 403,
    "origin_not_allowed": 403,
    "too_many_attempts": 429,
    "setup_required": 409,
    "last_admin": 409,
    "self_action_forbidden": 409,
    "username_taken": 409,
    "weak_password": 422,
    "current_password_incorrect": 422,
}


def test_every_code_has_status_and_message() -> None:
    for code in ErrorCode:
        assert code in ERROR_HTTP_STATUS and DEFAULT_MESSAGES[code].strip(), code
    for value, status in AUTH_CODES.items():
        assert ERROR_HTTP_STATUS[ErrorCode(value)] == status
    assert "make create-admin" in DEFAULT_MESSAGES[ErrorCode.SETUP_REQUIRED]


def test_pipeline_error_headers() -> None:
    err = PipelineError(ErrorCode.TOO_MANY_ATTEMPTS, headers={"Retry-After": "60"})
    assert (err.http_status, err.headers) == (429, {"Retry-After": "60"})
    assert PipelineError(ErrorCode.NOT_FOUND).headers == {}


# ---- settings --------------------------------------------------------------------------------


def test_session_settings_defaults(settings: Settings) -> None:
    assert (settings.session_idle_minutes, settings.session_absolute_hours) == (720, 168)
    assert (settings.session_idle_seconds, settings.session_absolute_seconds) == (
        12 * 3600,
        7 * 24 * 3600,
    )
    assert settings.cookie_secure is False


@pytest.mark.parametrize("origins", ["*", "http://localhost:3000, *"])
def test_wildcard_cors_origin_is_rejected(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, origins: str
) -> None:
    monkeypatch.setenv("MIMIC_CORS_ORIGINS", origins)
    get_settings.cache_clear()
    with pytest.raises(ValidationError, match="'\\*' is not allowed"):
        get_settings()


def test_cookie_secure_and_session_env(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MIMIC_COOKIE_SECURE", "1")
    monkeypatch.setenv("MIMIC_SESSION_IDLE_MINUTES", "30")
    monkeypatch.setenv("MIMIC_CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
    get_settings.cache_clear()
    s = get_settings()
    assert s.cookie_secure is True and s.session_idle_seconds == 1800
    assert s.cors_origins == ["http://localhost:3000", "http://127.0.0.1:3000"]
    monkeypatch.setenv("MIMIC_SESSION_ABSOLUTE_HOURS", "0")
    get_settings.cache_clear()
    with pytest.raises(ValidationError):
        get_settings()


# ---- policy ----------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["abc", "a.b", "user_1", "x-y-z", "0ab", "a" * 32])
def test_valid_usernames(name: str) -> None:
    assert policy.is_valid_username(name)


@pytest.mark.parametrize("name", ["ab", "Abc", "_abc", ".abc", "a b", "a" * 33, "ädmin", ""])
def test_invalid_usernames(name: str) -> None:
    assert not policy.is_valid_username(name)


def test_normalize_username() -> None:
    assert policy.normalize_username("  Alice \n") == "alice"


def test_password_policy() -> None:
    ok = "correct horse battery"
    assert policy.check_password_policy(ok, "alice") is None
    assert "at least 12" in (policy.check_password_policy("short", "alice") or "")
    assert "at most 256" in (policy.check_password_policy("x" * 257, "alice") or "")
    assert policy.check_password_policy("x" * 256, "alice") is None
    assert "username" in (policy.check_password_policy("Alice.Example", "alice.example") or "")
    assert "different" in (policy.check_password_policy(ok, "alice", current_password=ok) or "")
    assert policy.check_password_policy(ok, "alice", current_password=ok + "!") is None


def test_password_policy_counts_after_nfkc() -> None:
    # U+FB01 (ﬁ ligature) normalizes to "fi": 11 raw chars -> 12 after NFKC
    assert policy.check_password_policy("ﬁ" + "a" * 10, "alice") is None
    # full-width digits normalize to ASCII, so this equals the username
    assert policy.check_password_policy("１" * 12, "1" * 12) is not None


def test_scrypt_cost_is_lowered_for_tests() -> None:
    assert policy.current_scrypt_params().log2_n == 10
    assert policy.current_scrypt_params().maxmem == 64 * 1024 * 1024


@pytest.mark.real_scrypt
def test_real_scrypt_marker_keeps_production_cost() -> None:
    assert policy.current_scrypt_params() == policy.ScryptParams(log2_n=15, r=8, p=1, dklen=32)


def _user(role: Role, user_id: str = "u" * 32) -> UserRecord:
    now = datetime.now(UTC)
    return UserRecord(
        id=user_id,
        username="x",
        role=role,
        is_active=True,
        must_change_password=False,
        created_at=now,
        updated_at=now,
        password_changed_at=now,
        last_login_at=None,
        created_by=None,
        password_hash=UNUSABLE_PASSWORD_HASH,
    )


def test_can_access_job() -> None:
    admin, user = _user(Role.ADMIN, "a" * 32), _user(Role.USER, "b" * 32)
    assert policy.can_access_job(admin, None) and policy.can_access_job(admin, user.id)
    assert policy.can_access_job(user, user.id)
    assert not policy.can_access_job(user, admin.id)
    assert not policy.can_access_job(user, None)  # legacy job = admin-only


def test_user_record_repr_hides_password_hash() -> None:
    assert UNUSABLE_PASSWORD_HASH not in repr(_user(Role.USER))


# ---- models ----------------------------------------------------------------------------------


def test_login_request_normalizes_username() -> None:
    assert LoginRequest(username="  Alice ", password="p").username == "alice"
    with pytest.raises(ValidationError):
        LoginRequest(username="", password="p")
    with pytest.raises(ValidationError):
        LoginRequest.model_validate({"username": "a", "password": "p", "extra": 1})


def test_create_user_request() -> None:
    assert CreateUserRequest(username=" Bob.Smith ").model_dump() == {
        "username": "bob.smith",
        "role": "user",
    }
    for bad in ("ab", "bad name", "-lead"):
        with pytest.raises(ValidationError):
            CreateUserRequest(username=bad)
    with pytest.raises(ValidationError):
        CreateUserRequest.model_validate({"username": "carol", "role": "root"})


def test_update_user_request_needs_a_field() -> None:
    with pytest.raises(ValidationError, match="at least one"):
        UpdateUserRequest()
    with pytest.raises(ValidationError, match="at least one"):
        UpdateUserRequest.model_validate({"role": None, "is_active": None})
    assert UpdateUserRequest(is_active=False).role is None


def test_anonymous_health_and_owner_defaults() -> None:
    body = Health(version="1", ffmpeg=True).model_dump(mode="json")
    assert body == {
        "status": "ok",
        "version": "1",
        "ffmpeg": True,
        "interpreter": None,
        "limits": None,
    }
    assert "owner" in JobStatus.model_fields and JobStatus.model_fields["owner"].default is None


# ---- dependency chain ------------------------------------------------------------------------


def _deps_app() -> FastAPI:
    app = FastAPI()

    @app.exception_handler(PipelineError)
    async def _err(_: Request, exc: PipelineError) -> JSONResponse:
        return JSONResponse(status_code=exc.http_status, content=exc.to_body())

    @app.get("/session")
    def session(user: SessionUserDep) -> dict[str, str]:
        return {"id": user.id}

    @app.get("/current")
    def current(user: CurrentUserDep) -> dict[str, str]:
        return {"id": user.id}

    @app.get("/admin")
    def admin(user: AdminDep) -> dict[str, str]:
        return {"id": user.id}

    @app.get("/optional")
    def optional(user: OptionalUserDep) -> dict[str, str | None]:
        return {"id": user.id if user else None}

    return app


def _codes(client: TestClient) -> tuple[object, ...]:
    out: list[object] = []
    for path in ("/session", "/current", "/admin"):
        r = client.get(path)
        out.append(200 if r.status_code == 200 else r.json()["error"]["code"])
    out.append(client.get("/optional").json()["id"])
    return tuple(out)


def test_dependency_chain_with_as_user(tmp_path: Path) -> None:
    path = tmp_path / "mimic.db"
    admin = make_user(path, "admin", Role.ADMIN)
    user = make_user(path, "alice")
    forced = make_user(path, "newbie", must_change=True)
    disabled = make_user(path, "gone", active=False)
    forced_admin = make_user(path, "root2", Role.ADMIN, must_change=True)
    assert admin.is_admin and not user.is_admin and forced.must_change_password

    app = _deps_app()
    client = TestClient(app)
    expect = {
        None: ("unauthenticated",) * 3 + (None,),
        disabled: ("unauthenticated",) * 3 + (None,),
        forced: (200, "password_change_required", "password_change_required", None),
        forced_admin: (200, "password_change_required", "password_change_required", None),
        user: (200, 200, "forbidden", user.id),
        admin: (200, 200, 200, admin.id),
    }
    for who, codes in expect.items():
        with as_user(app, who):
            assert _codes(client) == codes, who
    assert app.dependency_overrides == {}  # restored


def test_as_user_nests_and_restores(tmp_path: Path) -> None:
    path = tmp_path / "mimic.db"
    alice, bob = make_user(path, "alice"), make_user(path, "bob")
    app = _deps_app()
    client = TestClient(app)
    with as_user(app, alice):
        with as_user(app, bob):
            assert client.get("/session").json()["id"] == bob.id
        assert client.get("/session").json()["id"] == alice.id
    assert app.dependency_overrides == {}


def test_deps_without_overrides(tmp_path: Path, settings: Settings) -> None:
    # B1 replaced the P0 stubs: real session resolution, still never an accidental pass.
    from app.auth.ratelimit import LoginLimiter
    from app.auth.sessions import SqliteSessionStore
    from app.auth.users import SqliteUserStore

    app = _deps_app()
    client = TestClient(app)
    # no app.state.auth (no lifespan): internal error, never an accidental pass
    assert client.get("/session").json()["error"]["code"] == "internal_error"
    path = tmp_path / "mimic.db"
    make_user(path, "alice")  # migrates
    app.state.auth = AuthServices(
        users=SqliteUserStore(path),
        sessions=SqliteSessionStore(path, idle_seconds=60, absolute_seconds=3600),
        limiter=LoginLimiter(),
        settings=settings,
    )
    client.cookies.set(SESSION_COOKIE, "anything")
    assert client.get("/session").json()["error"]["code"] == "unauthenticated"
    assert client.get("/admin").status_code == 401
    assert client.get("/optional").json() == {"id": None}
