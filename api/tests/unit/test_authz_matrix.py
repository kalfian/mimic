"""Authorization matrix with **real logins** (PLAN-auth §4, §9 INT, §11 P2, §14).

Every request is authenticated the way a browser is: ``POST /api/auth/login`` sets the
``mimic_session`` cookie and the cookie jar of each ``TestClient`` sends it. No dependency
overrides. One app (lifespan entered once) serves several clients, one cookie jar per actor.

Actors (``ACTORS``) are grouped by what the server must see (``Who``):

* ``NOBODY``: anonymous, and every dead session: disabled user (by an admin, and in the DB only),
  idle / absolute expiry (timestamps moved in the DB), revoked by logout (cookie replayed), role
  change, password reset, password change on another device, user deleted.
* ``FORCED``: user / admin with ``must_change_password``.
* ``OWNER`` (alice, owns the target job), ``OTHER`` (bob), ``ADMIN`` (root).

Endpoints (``ENDPOINTS``) carry their access rule (``Access``) and the status an allowed caller
gets. ``expected()`` is §4 as code; ``test_authz_matrix`` checks every actor × endpoint cell for
the exact status + error code, ``Cache-Control: no-store`` on auth/admin, 404 bodies identical
to an unknown id (non-disclosure), and that refused requests change nothing.

Further sections: Origin check on every unsafe route, cookie attributes, login outcomes,
expiry, last-admin guards incl. a concurrent mutual demotion/disable/delete race, user deletion
with a running job, legacy jobs + the first ``create-admin``, secrets in logs, import isolation.

Passwords here are test fixtures, not credentials.
"""

from __future__ import annotations

import ast
import hashlib
import io
import json
import logging
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from app import cli
from app.api.schemas import JobOptions, ResultEnvelope
from app.auth import policy as auth_policy
from app.auth.deps import SESSION_COOKIE
from app.auth.models import Role, UserRecord
from app.config import Settings
from app.core import db
from app.core.jobstore import SqliteJobStore
from app.core.runner import JobContext, ProgressReporter
from app.core.stages import Stage
from app.core.storage import LocalJobStorage
from app.main import create_app
from tests.auth_client import ADMIN_PW, USER_PW, create_account, error_code
from tests.conftest import SAMPLE_RESULT_PATH, TEST_SCRYPT_LOG2_N
from tests.unit import media_clips
from tests.unit.media_clips import requires_ffmpeg

API_DIR = Path(__file__).resolve().parents[2]
WRONG_PW = "definitely-not-the-password"
NEW_PW = "a fresh passphrase for tests"
EVIL_ORIGIN = "http://evil.localhost:5555"
#: 16 bytes, so a ``Range: bytes=0-3`` answer is ``bytes 0-3/16``.
PREVIEW_BYTES = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 4
PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16

#: username -> (password, role, create_account kwargs). ``root2`` is an *active* admin with a
#: forced change, so the last-admin tests disable it first.
ACCOUNTS: dict[str, tuple[str, Role, dict[str, bool]]] = {
    "root": (ADMIN_PW, Role.ADMIN, {}),
    "root2": (ADMIN_PW, Role.ADMIN, {"must_change": True}),
    "alice": (USER_PW, Role.USER, {}),
    "bob": (USER_PW, Role.USER, {}),
    "newbie": (USER_PW, Role.USER, {"must_change": True}),
    "dave": (USER_PW, Role.USER, {}),
    "gone": (USER_PW, Role.USER, {"active": False}),
}


def iso(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat(timespec="microseconds")


# --------------------------------------------------------------------------------------------
# fake pipeline + seeded jobs
# --------------------------------------------------------------------------------------------


def sample_result(job_id: str) -> dict[str, Any]:
    data = json.loads(SAMPLE_RESULT_PATH.read_text(encoding="utf-8"))
    data["job_id"] = data["spec"]["job_id"] = job_id
    data["artifacts"] = {"video_url": None, "keyframes": []}
    return data


def write_artifacts(storage: LocalJobStorage, job_id: str) -> None:
    """Preview, one keyframe and ``measurement.json`` (re-creates the job dir if needed)."""
    storage.artifact_path(job_id, "preview.mp4", create_parents=True).write_bytes(PREVIEW_BYTES)
    storage.artifact_path(job_id, "keyframes/state_a.png", create_parents=True).write_bytes(
        PNG_BYTES
    )
    storage.write_json(job_id, "measurement.json", {"format": -1})


def artifact_pipeline(ctx: JobContext, reporter: ProgressReporter) -> ResultEnvelope:
    reporter.enter(Stage.GENERATING)
    assert isinstance(ctx.storage, LocalJobStorage)
    write_artifacts(ctx.storage, ctx.job_id)
    return ResultEnvelope.model_validate(sample_result(ctx.job_id))


class Gate:
    """Wraps a pipeline so that its **first** run blocks until :attr:`release` is set."""

    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()

    def wrap(self, fn: Callable[[JobContext, ProgressReporter], ResultEnvelope]) -> Any:
        def run(ctx: JobContext, reporter: ProgressReporter) -> ResultEnvelope:
            if not self.started.is_set():
                reporter.enter(Stage.MEASURING, 0.5)
                self.started.set()
                assert self.release.wait(10)
                reporter.update(0.9)  # a write to a deleted row: a no-op
            return fn(ctx, reporter)

        return run


# --------------------------------------------------------------------------------------------
# the world: one app, real accounts, seeded jobs
# --------------------------------------------------------------------------------------------


def seed_job(store: SqliteJobStore, storage: LocalJobStorage, owner_id: str | None) -> str:
    """A ``succeeded`` job with input, result and every artifact."""
    job_id = uuid.uuid4().hex
    store.create(
        job_id, original_filename="clip.mp4", ext="mp4", options=JobOptions(), owner_id=owner_id
    )
    storage.artifact_path(job_id, "input.mp4", create_parents=True).write_bytes(b"\x00")
    storage.write_json(job_id, "result.json", sample_result(job_id))
    write_artifacts(storage, job_id)
    store.mark_succeeded(job_id)
    return job_id


@dataclass(frozen=True)
class Template:
    """A data dir with the ``ACCOUNTS`` users and four succeeded jobs with every artifact:
    ``legacy`` (no owner), ``root``, ``bob`` and ``alice`` (the matrix target). Built once per
    session and copied into every world (hashing + seeding per cell would dominate the run)."""

    data_dir: Path
    users: dict[str, UserRecord]
    jobs: dict[str, str]


@pytest.fixture(scope="session")
def template(tmp_path_factory: pytest.TempPathFactory) -> Template:
    data_dir = tmp_path_factory.mktemp("authz-template")
    db_path = data_dir / "mimic.db"
    with pytest.MonkeyPatch.context() as mp:  # the autouse fast-scrypt fixture is per test
        mp.setattr(
            auth_policy, "SCRYPT_PARAMS", auth_policy.ScryptParams(log2_n=TEST_SCRYPT_LOG2_N)
        )
        users = {
            name: create_account(db_path, name, pw, role, **kw)
            for name, (pw, role, kw) in ACCOUNTS.items()
        }
    store, storage = SqliteJobStore(db_path), LocalJobStorage(data_dir / "jobs")
    jobs = {
        key: seed_job(store, storage, users[key].id if key in users else None)
        for key in ("legacy", "root", "bob", "alice")
    }
    return Template(data_dir, users, jobs)


class World:
    """App with its lifespan entered on a copy of :class:`Template` (or, with ``base=None``, an
    empty data dir: a fresh / upgraded install without accounts)."""

    def __init__(
        self,
        settings: Settings,
        base: Template | None,
        *,
        pipeline: Any = artifact_pipeline,
        reinterpret: Any = artifact_pipeline,
    ) -> None:
        self.settings = settings
        self.db_path = settings.db_path
        self.store = SqliteJobStore(settings.db_path)
        self.storage = LocalJobStorage(settings.jobs_dir)
        self.users: dict[str, UserRecord] = {}
        self.jobs: dict[str, str] = {}
        if base is not None:
            settings.data_dir.mkdir(parents=True, exist_ok=True)
            src = sqlite3.connect(base.data_dir / "mimic.db")
            dst = sqlite3.connect(self.db_path)
            try:
                src.backup(dst)
            finally:
                src.close()
                dst.close()
            shutil.copytree(base.data_dir / "jobs", settings.jobs_dir, dirs_exist_ok=True)
            self.users, self.jobs = dict(base.users), dict(base.jobs)
        self.app = create_app(settings, pipeline=pipeline, reinterpret=reinterpret)
        self._life = TestClient(self.app)
        self._life.__enter__()
        self._ops: TestClient | None = None

    def close(self) -> None:
        self._life.__exit__(None, None, None)

    # ---- clients ----------------------------------------------------------------------------

    def client(self) -> TestClient:
        """A new cookie jar on the running app (the lifespan is already entered)."""
        return TestClient(self.app)

    def login(self, username: str, password: str | None = None) -> TestClient:
        c = self.client()
        pw = password if password is not None else ACCOUNTS[username][0]
        r = c.post("/api/auth/login", json={"username": username, "password": pw})
        assert r.status_code == 200, r.text
        return c

    @property
    def ops(self) -> TestClient:
        """root's own session for setup steps (separate from any actor's session)."""
        if self._ops is None:
            self._ops = self.login("root")
        return self._ops

    def call(
        self,
        client: TestClient,
        ep: Endpoint,
        *,
        job: str | None = None,
        origin: str | None = None,
    ) -> httpx.Response:
        path = ep.path.replace("{job}", job or self.jobs.get("alice", ""))
        if "{target}" in path:
            path = path.replace("{target}", self.users["bob"].id)
        headers = dict(ep.headers)
        if origin is not None:
            headers["Origin"] = origin
        files = dict(ep.files) if ep.files is not None else None
        return client.request(ep.method, path, json=ep.json, files=files, headers=headers)

    # ---- data -------------------------------------------------------------------------------

    def seed_job(self, owner_id: str | None) -> str:
        return seed_job(self.store, self.storage, owner_id)

    def sql(self, query: str, *params: Any) -> list[tuple[Any, ...]]:
        with db.connection(self.db_path) as conn:
            return [tuple(r) for r in conn.execute(query, params).fetchall()]

    def sessions(self, username: str | None = None) -> int:
        if username is None:
            return self.sql("SELECT count(*) FROM sessions")[0][0]
        uid = self.users[username].id
        return self.sql("SELECT count(*) FROM sessions WHERE user_id = ?", uid)[0][0]

    def active_admins(self) -> set[str]:
        rows = self.sql("SELECT username FROM users WHERE role = 'admin' AND is_active = 1")
        return {r[0] for r in rows}

    def snapshot(self) -> tuple[Any, ...]:
        """Users, jobs and job files (sessions excluded: resolving an expired one deletes it)."""
        users = self.sql(
            "SELECT id, username, role, is_active, must_change_password, password_hash "
            "FROM users ORDER BY id"
        )
        jobs = self.sql("SELECT id, status, owner_id, options_json FROM jobs ORDER BY id")
        root = self.settings.jobs_dir
        files = sorted(str(p.relative_to(root)) for p in root.rglob("*")) if root.exists() else []
        return users, jobs, files

    def job_gone(self, job_id: str) -> bool:
        return self.store.get(job_id) is None and not self.storage.exists(job_id)


WorldFactory = Callable[..., World]


@pytest.fixture
def make_world(settings: Settings, tmp_path: Path, template: Template) -> Iterator[WorldFactory]:
    """``make_world(settings_update=None, fresh=False, **World kwargs)``. The first world uses
    the ``settings`` fixture (so ``app.cli`` sees the same data dir); later ones get their own.
    ``fresh=True``: no accounts, no jobs."""
    opened: list[World] = []

    def factory(
        settings_update: dict[str, Any] | None = None, *, fresh: bool = False, **kw: Any
    ) -> World:
        update = dict(settings_update or {})
        if opened:
            update["data_dir"] = tmp_path / f"world{len(opened)}"
        s = settings.model_copy(update=update) if update else settings
        w = World(s, None if fresh else template, **kw)
        opened.append(w)
        return w

    yield factory
    for w in reversed(opened):
        w.close()


@pytest.fixture
def world(make_world: WorldFactory) -> World:
    return make_world()


# --------------------------------------------------------------------------------------------
# endpoints + access rules (§3.3, §4)
# --------------------------------------------------------------------------------------------


class Access(Enum):
    PUBLIC = "public"  # everyone (health: public subset without a full session)
    SESSION = "session"  # any live session, forced password change allowed
    FULL = "full"  # full session
    JOB = "job"  # full session + the target job is visible (else 404 not_found)
    ADMIN = "admin"  # full session + role admin


@dataclass(frozen=True, eq=False)
class Endpoint:
    name: str
    method: str
    #: ``{job}`` = target job id (alice's), ``{target}`` = bob's user id.
    path: str
    access: Access
    #: Status for a caller that passes authorization ...
    ok: int
    #: ... and its error code when the probe is designed to fail *after* authz (no side effects).
    ok_code: str | None = None
    json: Any = None
    files: Any = None
    headers: tuple[tuple[str, str], ...] = ()

    @property
    def unsafe(self) -> bool:
        return self.method not in ("GET", "HEAD", "OPTIONS")

    @property
    def no_store(self) -> bool:
        return self.path.startswith(("/api/auth/", "/api/admin/"))


ENDPOINTS: list[Endpoint] = [
    Endpoint("health", "GET", "/api/health", Access.PUBLIC, 200),
    Endpoint("auth-status", "GET", "/api/auth/status", Access.PUBLIC, 200),
    Endpoint(
        "auth-login",
        "POST",
        "/api/auth/login",
        Access.PUBLIC,
        200,
        json={"username": "bob", "password": USER_PW},
    ),
    Endpoint("auth-logout", "POST", "/api/auth/logout", Access.PUBLIC, 204),
    Endpoint("auth-me", "GET", "/api/auth/me", Access.SESSION, 200),
    Endpoint(
        "auth-password",
        "POST",
        "/api/auth/password",
        Access.SESSION,
        422,
        "current_password_incorrect",
        json={"current_password": WRONG_PW, "new_password": NEW_PW},
    ),
    Endpoint("admin-list", "GET", "/api/admin/users", Access.ADMIN, 200),
    Endpoint(
        "admin-create", "POST", "/api/admin/users", Access.ADMIN, 201, json={"username": "carol"}
    ),
    Endpoint(
        "admin-update",
        "PATCH",
        "/api/admin/users/{target}",
        Access.ADMIN,
        200,
        json={"is_active": False},
    ),
    Endpoint("admin-reset", "POST", "/api/admin/users/{target}/reset-password", Access.ADMIN, 200),
    Endpoint("admin-delete", "DELETE", "/api/admin/users/{target}", Access.ADMIN, 204),
    Endpoint(
        "jobs-create",
        "POST",
        "/api/jobs",
        Access.FULL,
        415,
        "unsupported_format",  # authz passed; the .txt is refused before anything is stored
        files=(("file", ("notes.txt", b"not a video", "text/plain")),),
    ),
    Endpoint("jobs-list", "GET", "/api/jobs", Access.FULL, 200),
    Endpoint("job-status", "GET", "/api/jobs/{job}", Access.JOB, 200),
    Endpoint("job-result", "GET", "/api/jobs/{job}/result", Access.JOB, 200),
    Endpoint("job-video", "GET", "/api/jobs/{job}/video", Access.JOB, 200),
    Endpoint(
        "job-video-range",
        "GET",
        "/api/jobs/{job}/video",
        Access.JOB,
        206,
        headers=(("Range", "bytes=0-3"),),
    ),
    Endpoint("job-keyframe", "GET", "/api/jobs/{job}/keyframes/state_a.png", Access.JOB, 200),
    Endpoint(
        "job-interpret",
        "POST",
        "/api/jobs/{job}/interpret",
        Access.JOB,
        202,
        json={"use_interpreter": False},
    ),
    Endpoint("job-delete", "DELETE", "/api/jobs/{job}", Access.JOB, 204),
    Endpoint("interpreter-check", "POST", "/api/interpreter/check", Access.ADMIN, 200),
]
UNSAFE_ENDPOINTS = [ep for ep in ENDPOINTS if ep.unsafe]
JOB_ENDPOINTS = [ep for ep in ENDPOINTS if ep.access is Access.JOB]


class Who(Enum):
    NOBODY = "no live session"
    FORCED = "forced password change"
    OWNER = "user owning the target job"
    OTHER = "user not owning the target job"
    ADMIN = "admin"


def expected(ep: Endpoint, who: Who) -> tuple[int, str | None]:
    """PLAN-auth §4. Check order: session (401) → forced change (403) → role (403) → job (404)."""
    allowed = (ep.ok, ep.ok_code)
    if ep.access is Access.PUBLIC:
        return allowed
    if who is Who.NOBODY:
        return 401, "unauthenticated"
    if ep.access is Access.SESSION:
        return allowed
    if who is Who.FORCED:
        return 403, "password_change_required"
    if ep.access is Access.ADMIN and who is not Who.ADMIN:
        return 403, "forbidden"
    if ep.access is Access.JOB and who is Who.OTHER:
        return 404, "not_found"
    return allowed


# --------------------------------------------------------------------------------------------
# actors
# --------------------------------------------------------------------------------------------


def _alice_id(w: World) -> str:
    return w.users["alice"].id


def _disabled_by_admin(w: World) -> TestClient:
    c = w.login("alice")
    r = w.ops.patch(f"/api/admin/users/{_alice_id(w)}", json={"is_active": False})
    assert r.status_code == 200, r.text
    assert w.sessions("alice") == 0  # revoked immediately (U2)
    return c


def _disabled_in_db_only(w: World) -> TestClient:
    """Defence in depth: the session row survives, resolving it must still fail (§4)."""
    c = w.login("alice")
    w.sql("UPDATE users SET is_active = 0 WHERE id = ?", _alice_id(w))
    assert w.sessions("alice") == 1
    return c


def _expire(w: World, username: str, kind: str) -> None:
    now = datetime.now(UTC)
    uid = w.users[username].id
    if kind == "idle":
        stale = now - timedelta(seconds=w.settings.session_idle_seconds) - timedelta(minutes=1)
        w.sql("UPDATE sessions SET last_seen_at = ? WHERE user_id = ?", iso(stale), uid)
    else:  # absolute: last_seen is fresh, expires_at has passed
        w.sql("UPDATE sessions SET expires_at = ? WHERE user_id = ?", iso(now - timedelta(1)), uid)


def _expired(kind: str) -> Callable[[World], TestClient]:
    def build(w: World) -> TestClient:
        c = w.login("alice")
        _expire(w, "alice", kind)
        return c

    return build


def _logged_out_replayed(w: World) -> TestClient:
    c = w.login("alice")
    token = c.cookies.get(SESSION_COOKIE)
    assert token and c.post("/api/auth/logout").status_code == 204
    assert w.sessions("alice") == 0
    c.cookies.set(SESSION_COOKIE, token)  # replay the old cookie
    return c


def _role_changed(w: World) -> TestClient:
    c = w.login("alice")
    r = w.ops.patch(f"/api/admin/users/{_alice_id(w)}", json={"role": "admin"})
    assert r.status_code == 200, r.text
    assert w.sessions("alice") == 0
    return c


def _password_reset(w: World) -> TestClient:
    c = w.login("alice")
    assert w.ops.post(f"/api/admin/users/{_alice_id(w)}/reset-password").status_code == 200
    assert w.sessions("alice") == 0
    return c


def _password_changed_elsewhere(w: World) -> TestClient:
    c = w.login("alice")
    other = w.login("alice")
    r = other.post("/api/auth/password", json={"current_password": USER_PW, "new_password": NEW_PW})
    assert r.status_code == 200, r.text
    assert w.sessions("alice") == 1  # only the new session of ``other``
    return c


def _user_deleted(w: World) -> TestClient:
    c = w.login("dave")
    assert w.ops.delete(f"/api/admin/users/{w.users['dave'].id}").status_code == 204
    assert w.sessions("dave") == 0
    return c


@dataclass(frozen=True, eq=False)
class Actor:
    name: str
    who: Who
    build: Callable[[World], TestClient]
    #: Account behind a live session (for ``/me`` and the job list).
    username: str | None = None


def _as(username: str) -> Callable[[World], TestClient]:
    return lambda w: w.login(username)


ACTORS: list[Actor] = [
    Actor("anonymous", Who.NOBODY, lambda w: w.client()),
    Actor("admin", Who.ADMIN, _as("root"), "root"),
    Actor("user_a", Who.OWNER, _as("alice"), "alice"),
    Actor("user_b", Who.OTHER, _as("bob"), "bob"),
    Actor("forced_user", Who.FORCED, _as("newbie"), "newbie"),
    Actor("forced_admin", Who.FORCED, _as("root2"), "root2"),
    Actor("disabled", Who.NOBODY, _disabled_by_admin),
    Actor("disabled_db_only", Who.NOBODY, _disabled_in_db_only),
    Actor("expired_idle", Who.NOBODY, _expired("idle")),
    Actor("expired_absolute", Who.NOBODY, _expired("absolute")),
    Actor("revoked_logout", Who.NOBODY, _logged_out_replayed),
    Actor("revoked_role_change", Who.NOBODY, _role_changed),
    Actor("revoked_password_reset", Who.NOBODY, _password_reset),
    Actor("revoked_password_change", Who.NOBODY, _password_changed_elsewhere),
    Actor("deleted_user", Who.NOBODY, _user_deleted),
]


# --------------------------------------------------------------------------------------------
# the matrix
# --------------------------------------------------------------------------------------------


def _check_success(w: World, client: TestClient, actor: Actor, ep: Endpoint, r: Any) -> None:
    """Endpoint-specific assertions for an allowed call."""
    alice_job = w.jobs["alice"]
    if ep.name == "health":
        body = r.json()
        full = actor.who not in (Who.NOBODY, Who.FORCED)
        assert (body["interpreter"] is not None, body["limits"] is not None) == (full, full)
        assert body["ffmpeg"] in (True, False) and body["version"]
    elif ep.name == "auth-login":
        assert r.json()["username"] == "bob" and SESSION_COOKIE in r.headers["set-cookie"]
    elif ep.name == "auth-logout":
        assert "max-age=0" in r.headers["set-cookie"].lower()
        assert client.get("/api/auth/me").status_code == 401
    elif ep.name == "auth-me":
        assert r.json()["username"] == actor.username
    elif ep.name == "admin-create":
        body = r.json()
        assert body["user"]["must_change_password"] is True and body["temporary_password"]
    elif ep.name == "admin-update":
        assert r.json()["is_active"] is False
    elif ep.name == "admin-delete":
        assert w.job_gone(w.jobs["bob"]) and not w.job_gone(alice_job)
    elif ep.name == "jobs-list":
        listed = {j["id"] for j in r.json()["items"]}
        if actor.who is Who.ADMIN:
            assert listed == set(w.jobs.values())
        else:
            assert listed == {w.jobs[str(actor.username)]}
    elif ep.name in ("job-status", "job-interpret"):
        assert r.json()["owner"] == {"id": _alice_id(w), "username": "alice"}
    elif ep.name == "job-video":
        assert r.content == PREVIEW_BYTES and r.headers["content-type"] == "video/mp4"
    elif ep.name == "job-video-range":
        assert r.headers["content-range"] == f"bytes 0-3/{len(PREVIEW_BYTES)}"
        assert r.content == PREVIEW_BYTES[:4]
    elif ep.name == "job-keyframe":
        assert r.content == PNG_BYTES
    elif ep.name == "job-delete":
        assert w.job_gone(alice_job)
    elif ep.name == "interpreter-check":
        assert r.json()["mode"] == "none"


@pytest.mark.parametrize("ep", ENDPOINTS, ids=lambda e: e.name)
@pytest.mark.parametrize("actor", ACTORS, ids=lambda a: a.name)
def test_authz_matrix(world: World, actor: Actor, ep: Endpoint) -> None:
    client = actor.build(world)
    before = world.snapshot()
    r = world.call(client, ep)

    status, code = expected(ep, actor.who)
    assert r.status_code == status, r.text
    if code is not None:
        assert error_code(r) == code
    if ep.no_store:
        assert r.headers.get("cache-control") == "no-store"
    if status < 400:
        _check_success(world, client, actor, ep, r)
        return
    if status == 404:  # non-disclosure: exactly what an unknown id gets
        unknown = world.call(client, ep, job=uuid.uuid4().hex)
        assert unknown.status_code == 404
        assert unknown.json() == r.json()
        assert unknown.headers["content-type"] == r.headers["content-type"]
    assert "set-cookie" not in r.headers or ep.name == "auth-password"
    assert world.snapshot() == before, "a refused request changed state"


@pytest.mark.parametrize("target", ["legacy", "root", "bob"])
def test_foreign_and_legacy_jobs_look_like_unknown_ids(world: World, target: str) -> None:
    """alice gets 404 on every route of a job she doesn't own (legacy = admin-only, A10),
    indistinguishable from an id that doesn't exist; an admin may use it."""
    job = world.jobs[target]
    alice = world.login("alice")
    before = world.snapshot()
    for ep in JOB_ENDPOINTS:
        r = world.call(alice, ep, job=job)
        unknown = world.call(alice, ep, job=uuid.uuid4().hex)
        assert (r.status_code, error_code(r)) == (404, "not_found"), ep.name
        assert r.json() == unknown.json(), ep.name
    assert world.snapshot() == before

    admin = world.login("root")
    owner = admin.get(f"/api/jobs/{job}").json()["owner"]
    assert owner == (None if target == "legacy" else {"id": world.users[target].id,
                                                      "username": target})  # fmt: skip
    for ep in JOB_ENDPOINTS:
        if not ep.unsafe:
            assert world.call(admin, ep, job=job).status_code == ep.ok, ep.name


def test_job_list_owner_filter(world: World) -> None:
    ids = world.jobs

    def listed(c: TestClient, **params: str) -> set[str] | int:
        r = c.get("/api/jobs", params=params)
        return {j["id"] for j in r.json()["items"]} if r.status_code == 200 else r.status_code

    admin = world.login("root")
    assert listed(admin) == set(ids.values())
    assert listed(admin, owner="me") == {ids["root"]}
    assert listed(admin, owner=world.users["alice"].id) == {ids["alice"]}
    items = {j["id"]: j["owner"] for j in admin.get("/api/jobs").json()["items"]}
    assert items[ids["legacy"]] is None and items[ids["bob"]]["username"] == "bob"
    for name, other in (("alice", "bob"), ("bob", "alice")):
        c = world.login(name)
        own = {ids[name]}
        assert listed(c) == own == listed(c, owner="me") == listed(c, owner=world.users[name].id)
        for foreign in (world.users[other].id, world.users["root"].id):
            r = c.get("/api/jobs", params={"owner": foreign})
            assert (r.status_code, error_code(r)) == (403, "forbidden")


def test_implicit_head_is_not_a_bypass(world: World) -> None:
    anon = world.client()
    for path in ("", "/result", "/video", "/keyframes/state_a.png"):
        assert anon.head(f"/api/jobs/{world.jobs['alice']}{path}").status_code in (401, 405)


# --------------------------------------------------------------------------------------------
# Origin check (A7) on every unsafe route
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("origin", [EVIL_ORIGIN, "http://localhost:5555", "null"])
@pytest.mark.parametrize("ep", UNSAFE_ENDPOINTS, ids=lambda e: e.name)
def test_foreign_origin_is_refused_first(world: World, ep: Endpoint, origin: str) -> None:
    """403 ``origin_not_allowed`` before authentication (anonymous gets 403, not 401), with no
    side effect: no cookie set, no session revoked, nothing written."""
    clients = {"anonymous": world.client(), "admin": world.login("root")}
    for who, client in clients.items():
        before, sessions = world.snapshot(), world.sessions()
        r = world.call(client, ep, origin=origin)
        assert (r.status_code, error_code(r)) == (403, "origin_not_allowed"), who
        assert r.headers["cache-control"] == "no-store"
        assert "set-cookie" not in r.headers
        assert (world.snapshot(), world.sessions()) == (before, sessions), who
    assert clients["admin"].get("/api/auth/me").status_code == 200  # e.g. no forged logout


@pytest.mark.parametrize("origin", ["allowed", "own", "absent"])
@pytest.mark.parametrize("ep", UNSAFE_ENDPOINTS, ids=lambda e: e.name)
def test_trusted_or_missing_origin_passes(world: World, ep: Endpoint, origin: str) -> None:
    value = {
        "allowed": world.settings.cors_origins[0],  # MIMIC_CORS_ORIGINS
        "own": "http://testserver",  # the API's own origin (Swagger /docs)
        "absent": None,  # non-browser client
    }[origin]
    r = world.call(world.login("root"), ep, origin=value)
    status, code = expected(ep, Who.ADMIN)
    assert r.status_code == status, r.text
    if code is not None:
        assert error_code(r) == code
    if origin == "allowed":
        assert r.headers["access-control-allow-origin"] == value
        assert r.headers["access-control-allow-credentials"] == "true"


# --------------------------------------------------------------------------------------------
# cookie (A2) + login outcomes
# --------------------------------------------------------------------------------------------


def _cookie(header: str) -> tuple[str, str, dict[str, str]]:
    first, *attrs = (p.strip() for p in header.split(";"))
    name, _, value = first.partition("=")
    parsed = {}
    for a in attrs:
        k, _, v = a.partition("=")
        parsed[k.lower()] = v.lower()
    return name, value.strip('"'), parsed


@pytest.mark.parametrize("secure", [False, True], ids=["http", "cookie_secure"])
def test_session_cookie_attributes(make_world: WorldFactory, secure: bool) -> None:
    w = make_world(settings_update={"cookie_secure": secure})
    c = w.client()

    def check(header: str, *, cleared: bool) -> str:
        name, value, attrs = _cookie(header)
        assert name == SESSION_COOKIE
        assert "httponly" in attrs and attrs["samesite"] == "lax" and attrs["path"] == "/"
        assert "domain" not in attrs  # host-only
        assert ("secure" in attrs) is secure
        max_age = "0" if cleared else str(w.settings.session_absolute_seconds)
        assert attrs["max-age"] == max_age
        assert bool(value) is not cleared
        return value

    token = check(
        c.post("/api/auth/login", json={"username": "alice", "password": USER_PW}).headers[
            "set-cookie"
        ],
        cleared=False,
    )
    # only sha256(token) is stored
    stored = {r[0] for r in w.sql("SELECT token_hash FROM sessions")}
    assert hashlib.sha256(token.encode()).hexdigest() in stored and token not in stored

    # a Secure cookie isn't sent over http by the client jar: present it explicitly
    auth = {"Cookie": f"{SESSION_COOKIE}={token}"}
    plain = w.client()
    r = plain.post(
        "/api/auth/password",
        json={"current_password": USER_PW, "new_password": NEW_PW},
        headers=auth,
    )
    assert r.status_code == 200, r.text
    renewed = check(r.headers["set-cookie"], cleared=False)
    assert renewed != token

    r = w.client().post("/api/auth/logout", headers={"Cookie": f"{SESSION_COOKIE}={renewed}"})
    assert r.status_code == 204
    check(r.headers["set-cookie"], cleared=True)
    assert w.sessions("alice") == 0


LOGIN_CASES = [
    # (case, username, password, status, error code, must_change_password)
    ("admin", "root", ADMIN_PW, 200, None, False),
    ("user", "alice", USER_PW, 200, None, False),
    ("user_name_normalized", "  ALICE ", USER_PW, 200, None, False),
    ("forced_user", "newbie", USER_PW, 200, None, True),
    ("forced_admin", "root2", ADMIN_PW, 200, None, True),
    ("wrong_password", "alice", WRONG_PW, 401, "invalid_credentials", None),
    ("unknown_user", "nobody", USER_PW, 401, "invalid_credentials", None),
    ("disabled_wrong_password", "gone", WRONG_PW, 401, "invalid_credentials", None),
    ("disabled_right_password", "gone", USER_PW, 403, "account_disabled", None),
]


@pytest.mark.parametrize(
    ("username", "password", "status", "code", "must_change"),
    [c[1:] for c in LOGIN_CASES],
    ids=[c[0] for c in LOGIN_CASES],
)
def test_login_outcomes(
    world: World,
    username: str,
    password: str,
    status: int,
    code: str | None,
    must_change: bool | None,
) -> None:
    c = world.client()
    r = c.post("/api/auth/login", json={"username": username, "password": password})
    assert r.status_code == status, r.text
    assert r.headers["cache-control"] == "no-store"
    if code is None:
        assert r.json()["must_change_password"] is must_change
        assert c.get("/api/auth/me").json()["username"] == username.strip().lower()
        return
    assert error_code(r) == code and "set-cookie" not in r.headers
    if code == "invalid_credentials":  # no user enumeration: same body for every failure
        unknown = c.post("/api/auth/login", json={"username": "nobody2", "password": WRONG_PW})
        assert unknown.json() == r.json()


def test_deleted_user_cannot_sign_in(world: World) -> None:
    assert world.ops.delete(f"/api/admin/users/{world.users['dave'].id}").status_code == 204
    r = world.client().post("/api/auth/login", json={"username": "dave", "password": USER_PW})
    assert (r.status_code, error_code(r)) == (401, "invalid_credentials")


# --------------------------------------------------------------------------------------------
# expiry (§2, §4: 401 + row deleted)
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["idle", "absolute"])
def test_expired_session_is_refused_and_deleted(world: World, kind: str) -> None:
    c = world.login("alice")
    _expire(world, "alice", kind)
    assert world.sessions("alice") == 1
    r = c.get(f"/api/jobs/{world.jobs['alice']}")
    assert (r.status_code, error_code(r)) == (401, "unauthenticated")
    assert world.sessions("alice") == 0
    assert c.get("/api/health").json()["limits"] is None  # public subset only


def test_session_inside_idle_window_is_kept_and_touched(world: World) -> None:
    c = world.login("alice")
    idle = timedelta(seconds=world.settings.session_idle_seconds)
    almost = datetime.now(UTC) - idle + timedelta(minutes=5)
    world.sql("UPDATE sessions SET last_seen_at = ?", iso(almost))
    assert c.get("/api/auth/me").status_code == 200
    (seen,) = world.sql("SELECT last_seen_at FROM sessions")[0]
    assert datetime.fromisoformat(seen) > datetime.now(UTC) - timedelta(minutes=1)


# --------------------------------------------------------------------------------------------
# last active admin (U2, A13)
# --------------------------------------------------------------------------------------------


def test_last_admin_and_self_guards(world: World) -> None:
    root, root2 = world.users["root"], world.users["root2"]
    ops = world.ops
    # root2 is an active admin (forced change): disable it so root is the last active admin
    assert ops.patch(f"/api/admin/users/{root2.id}", json={"is_active": False}).status_code == 200
    assert world.active_admins() == {"root"}
    before = world.snapshot()
    for method, path, body, code in [
        ("PATCH", f"/api/admin/users/{root.id}", {"role": "user"}, "last_admin"),
        ("PATCH", f"/api/admin/users/{root.id}", {"is_active": False}, "self_action_forbidden"),
        ("POST", f"/api/admin/users/{root.id}/reset-password", None, "self_action_forbidden"),
        ("DELETE", f"/api/admin/users/{root.id}", None, "self_action_forbidden"),
    ]:
        r = ops.request(method, path, json=body)
        assert (r.status_code, error_code(r)) == (409, code), (method, body)
    assert world.snapshot() == before
    assert ops.get("/api/admin/users").status_code == 200  # refused actions revoked nothing

    # a disabled admin doesn't count; once root2 is enabled root may demote itself
    assert ops.patch(f"/api/admin/users/{root2.id}", json={"is_active": True}).status_code == 200
    r = ops.patch(f"/api/admin/users/{root.id}", json={"role": "user"})
    assert r.status_code == 200 and r.json()["role"] == "user"
    assert ops.get("/api/auth/me").status_code == 401  # own sessions revoked by the role change
    assert world.active_admins() == {"root2"}


def _race_request(client: TestClient, action: str, target_id: str) -> httpx.Response:
    path = f"/api/admin/users/{target_id}"
    if action == "demote":
        return client.patch(path, json={"role": "user"})
    if action == "disable":
        return client.patch(path, json={"is_active": False})
    return client.delete(path)


def _race_round(w: World, action: str) -> None:
    create_account(w.db_path, "root3", ADMIN_PW, Role.ADMIN)
    r2 = w.users["root2"].id
    assert w.ops.patch(f"/api/admin/users/{r2}", json={"is_active": False}).status_code == 200
    assert w.active_admins() == {"root", "root3"}
    ids = {n: w.sql("SELECT id FROM users WHERE username = ?", n)[0][0] for n in ("root", "root3")}
    clients = {name: w.login(name, ADMIN_PW) for name in ids}
    barrier = threading.Barrier(2)
    results: dict[str, httpx.Response] = {}

    def act(name: str, other: str) -> None:
        barrier.wait()
        results[name] = _race_request(clients[name], action, ids[other])

    threads = [
        threading.Thread(target=act, args=("root", "root3")),
        threading.Thread(target=act, args=("root3", "root")),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(15)
    assert set(results) == {"root", "root3"}

    winners = [n for n, r in results.items() if r.status_code in (200, 204)]
    assert len(winners) == 1, {n: (r.status_code, r.text) for n, r in results.items()}
    (winner,) = winners
    lost = next(r for n, r in results.items() if n != winner)
    assert (lost.status_code, error_code(lost)) in {
        (409, "last_admin"),
        (401, "unauthenticated"),
        (403, "forbidden"),
    }
    assert w.active_admins() == {winner}
    assert clients[winner].get("/api/admin/users").status_code == 200


RACE_ROUNDS = 5


@pytest.mark.parametrize("action", ["demote", "disable", "delete"])
def test_concurrent_mutual_last_admin_race(make_world: WorldFactory, action: str) -> None:
    """Two active admins act on each other at the same moment: exactly one wins and one active
    admin is left. The loser is refused by the atomic guard (409 ``last_admin``) or, if the
    winner's change landed before the loser's request was authorized, by its revoked session
    (401) or its fresh role (403)."""
    for _ in range(RACE_ROUNDS):
        _race_round(make_world(), action)


# --------------------------------------------------------------------------------------------
# user deletion with a running job (A14) — no orphan dir
# --------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return media_clips.testsrc(
        tmp_path_factory.mktemp("authz-clip") / "ok.mp4", duration=2.0, size="320x240"
    )


def _wait_succeeded(client: TestClient, job_id: str, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while (body := client.get(f"/api/jobs/{job_id}").json())["status"] != "succeeded":
        assert body["status"] != "failed", body
        assert time.monotonic() < deadline, body
        time.sleep(0.02)


@pytest.mark.parametrize("kind", [pytest.param("upload", marks=requires_ffmpeg), "rerun"])
def test_delete_user_with_running_job_leaves_no_orphans(
    make_world: WorldFactory, request: pytest.FixtureRequest, kind: str
) -> None:
    gate = Gate()
    w = make_world(pipeline=gate.wrap(artifact_pipeline), reinterpret=gate.wrap(artifact_pipeline))
    assert w.settings.workers == 1  # the follow-up job below runs after the deleted one returned
    alice = w.login("alice")
    done = w.jobs["alice"]
    if kind == "upload":
        with request.getfixturevalue("clip").open("rb") as f:
            r = alice.post("/api/jobs", files={"file": ("ok.mp4", f, "video/mp4")})
        assert r.status_code == 202, r.text
        running = r.json()["id"]
        alice_jobs = [done, running]
    else:
        r = alice.post(f"/api/jobs/{done}/interpret", json={"use_interpreter": False})
        assert r.status_code == 202, r.text
        running = done
        alice_jobs = [done]
    assert gate.started.wait(10)
    assert alice.get(f"/api/jobs/{running}").json()["status"] == "processing"

    r = w.ops.delete(f"/api/admin/users/{_alice_id(w)}")
    assert r.status_code == 204
    assert all(w.job_gone(j) for j in alice_jobs)
    assert alice.get("/api/auth/me").status_code == 401

    gate.release.set()  # the pipeline now writes artifacts again (re-creating the dir)
    root_job = w.jobs["root"]
    assert (
        w.ops.post(f"/api/jobs/{root_job}/interpret", json={"use_interpreter": False}).status_code
        == 202
    )
    _wait_succeeded(w.ops, root_job)

    assert all(w.job_gone(j) for j in alice_jobs), "the runner left an orphan dir or row"
    assert set(w.storage.job_ids()) == {w.jobs["legacy"], root_job, w.jobs["bob"]}
    listed = {j["id"] for j in w.ops.get("/api/jobs").json()["items"]}
    assert listed == {w.jobs["legacy"], root_job, w.jobs["bob"]}
    for job in alice_jobs:
        assert w.ops.get(f"/api/jobs/{job}").status_code == 404


def test_upload_racing_its_owners_deletion_leaves_nothing(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression (U2/A14): an admin deletes alice while her upload is still being validated
    (ffprobe + first-frame decode take a while). The job row used to be inserted afterwards
    with the dangling ``owner_id`` (``jobs.owner_id`` has no FK): her video survived her
    deletion, invisible to everyone but admins. Now the insert is refused, the upload is
    removed and she gets 401 like any request after her account is gone."""
    from app.api import routes_jobs

    alice_id = _alice_id(world)
    deleted: list[int] = []

    def validate_then_owner_deleted(path: Path, settings: Settings) -> Any:
        deleted.append(world.ops.delete(f"/api/admin/users/{alice_id}").status_code)
        return SimpleNamespace(width=320, height=240, fps_nominal=30.0, duration_s=2.0)

    monkeypatch.setattr(routes_jobs, "validate_upload", validate_then_owner_deleted)
    alice = world.login("alice")
    r = alice.post("/api/jobs", files={"file": ("clip.mp4", b"\x00" * 64, "video/mp4")})
    assert deleted == [204]
    assert (r.status_code, error_code(r)) == (401, "unauthenticated")
    assert world.sql("SELECT count(*) FROM jobs WHERE owner_id = ?", alice_id) == [(0,)]
    expected = {world.jobs[k] for k in ("legacy", "root", "bob")}
    assert set(world.storage.job_ids()) == expected  # no dir of the refused upload


# --------------------------------------------------------------------------------------------
# legacy jobs + first create-admin (A10, A11)
# --------------------------------------------------------------------------------------------


def test_legacy_jobs_claimed_by_first_create_admin(
    make_world: WorldFactory, capsys: pytest.CaptureFixture[str]
) -> None:
    w = make_world(fresh=True)  # an upgraded pre-auth install
    legacy = {w.seed_job(None), w.seed_job(None)}
    anon = w.client()
    assert anon.get("/api/auth/status").json() == {"setup_required": True}
    r = anon.post("/api/auth/login", json={"username": "root", "password": ADMIN_PW})
    assert (r.status_code, error_code(r)) == (409, "setup_required")
    some = next(iter(legacy))
    r = anon.get(f"/api/jobs/{some}")
    assert (r.status_code, error_code(r)) == (401, "unauthenticated")

    def create_admin(name: str) -> str:
        argv = ["create-admin", "--username", name, "--password-stdin"]
        assert cli.main(argv, stdin=io.StringIO(ADMIN_PW + "\n")) == 0
        return capsys.readouterr().out

    assert "Assigned 2 existing jobs to root" in create_admin("root")
    assert anon.get("/api/auth/status").json() == {"setup_required": False}
    root = w.login("root", ADMIN_PW)
    items = root.get("/api/jobs").json()["items"]
    assert {j["id"] for j in items} == legacy
    assert {j["owner"]["username"] for j in items} == {"root"}

    # an orphan appearing later (old code) stays admin-only; a second admin claims nothing
    later = w.seed_job(None)
    assert "Assigned" not in create_admin("root2")
    assert root.get(f"/api/jobs/{later}").json()["owner"] is None
    root2 = w.login("root2", ADMIN_PW)
    assert root2.get(f"/api/jobs/{later}").status_code == 200

    temp = root.post("/api/admin/users", json={"username": "alice"}).json()["temporary_password"]
    alice = w.login("alice", temp)
    r = alice.post("/api/auth/password", json={"current_password": temp, "new_password": NEW_PW})
    assert r.status_code == 200
    for job in (later, some):
        for ep in JOB_ENDPOINTS:
            r = w.call(alice, ep, job=job)
            assert (r.status_code, error_code(r)) == (404, "not_found"), ep.name
    assert alice.get("/api/jobs").json()["items"] == []


# --------------------------------------------------------------------------------------------
# secrets never reach the logs (§5)
# --------------------------------------------------------------------------------------------


def test_no_password_token_or_temp_password_in_logs(
    world: World, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    caplog.set_level(logging.DEBUG, logger="app")
    tokens: list[str] = []

    def keep_token(c: TestClient) -> TestClient:
        tok = c.cookies.get(SESSION_COOKIE)
        assert tok
        tokens.append(tok)
        return c

    anon = world.client()
    for _ in range(2):
        anon.post("/api/auth/login", json={"username": "alice", "password": WRONG_PW})
    root = keep_token(world.login("root"))
    created = root.post("/api/admin/users", json={"username": "carol"}).json()
    temp1, carol_id = created["temporary_password"], created["user"]["id"]
    carol = keep_token(world.login("carol", temp1))
    r = carol.post("/api/auth/password", json={"current_password": temp1, "new_password": NEW_PW})
    assert r.status_code == 200
    keep_token(carol)
    temp2 = root.post(f"/api/admin/users/{carol_id}/reset-password").json()["temporary_password"]
    carol = keep_token(world.login("carol", temp2))
    carol.post("/api/auth/password", json={"current_password": WRONG_PW, "new_password": NEW_PW})
    carol.post("/api/auth/logout")
    alice = keep_token(world.login("alice"))
    alice.delete(f"/api/jobs/{world.jobs['alice']}")
    assert root.patch(f"/api/admin/users/{carol_id}", json={"is_active": False}).status_code == 200
    assert root.delete(f"/api/admin/users/{_alice_id(world)}").status_code == 204

    hashes = [r[0] for r in world.sql("SELECT password_hash FROM users")]
    secrets = [ADMIN_PW, USER_PW, WRONG_PW, NEW_PW, temp1, temp2, *tokens, *hashes]
    secrets += [hashlib.sha256(t.encode()).hexdigest() for t in tokens]
    text = caplog.text
    leaked = [i for i, s in enumerate(secrets) if s in text]  # indexes only: never print them
    assert leaked == []
    # the audit lines were captured, so the check above is not vacuous
    assert "signed in" in text and "failed sign-in" in text
    assert f"reset the password of user {carol_id}" in text
    assert f"deleted user {_alice_id(world)}" in text


# --------------------------------------------------------------------------------------------
# the pipeline and scripts stay auth-free (§1, §9)
# --------------------------------------------------------------------------------------------


def _absolute_imports(path: Path) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.add(node.module)
            found |= {f"{node.module}.{a.name}" for a in node.names}
    return found


def test_pipeline_and_scripts_never_import_auth() -> None:
    files = sorted((API_DIR / "app" / "pipeline").rglob("*.py"))
    files += sorted((API_DIR / "scripts").glob("*.py"))
    assert len(files) > 5
    offenders = {
        str(f.relative_to(API_DIR)): sorted(mods)
        for f in files
        if (mods := {m for m in _absolute_imports(f) if m.split(".")[:2] == ["app", "auth"]})
    }
    assert offenders == {}


def test_pipeline_loads_only_the_pure_auth_policy() -> None:
    """Transitively (``app.api.schemas`` mirrors the username pattern) only the I/O-free
    ``app.auth.policy`` is loaded: no sessions, users, passwords, deps or middleware."""
    code = (
        "import json, sys\n"
        "import app.config, app.core.errors, app.generate, app.pipeline.run\n"
        "print(json.dumps(sorted(m for m in sys.modules if m.startswith('app.auth'))))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=API_DIR, capture_output=True, text=True, check=True
    ).stdout
    assert set(json.loads(out.strip().splitlines()[-1])) <= {"app.auth", "app.auth.policy"}
