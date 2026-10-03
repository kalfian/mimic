"""SQLite user store (PLAN-auth §2, U2, A10, A17)."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from app.api.schemas import JobOptions
from app.auth.models import Role
from app.auth.sessions import SqliteSessionStore
from app.auth.users import SqliteUserStore
from app.core import db
from app.core.errors import ErrorCode, PipelineError
from app.core.jobstore import SqliteJobStore

HASH = "scrypt$1$1$1$AAAAAAAAAAAAAAAAAAAAAA==$" + "A" * 44  # never verified here


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    path = tmp_path / "mimic.db"
    db.migrate(path)
    return path


@pytest.fixture
def users(db_path: Path) -> SqliteUserStore:
    return SqliteUserStore(db_path)


def _code(exc: pytest.ExceptionInfo[PipelineError]) -> ErrorCode:
    return exc.value.code


def _job(db_path: Path, job_id: str, owner_id: str | None) -> None:
    SqliteJobStore(db_path).create(
        job_id, original_filename="a.mp4", ext="mp4", options=JobOptions(), owner_id=owner_id
    )


def test_create_normalizes_and_is_unique(users: SqliteUserStore) -> None:
    u = users.create(" Alice ", HASH, Role.USER, must_change_password=True, created_by="x" * 32)
    assert (u.username, u.role, u.is_active, u.must_change_password) == (
        "alice",
        Role.USER,
        True,
        True,
    )
    assert u.created_by == "x" * 32 and u.last_login_at is None and len(u.id) == 32
    assert users.get(u.id) == u
    assert users.get_by_username("ALICE") == u
    with pytest.raises(PipelineError) as exc:
        users.create("alice", HASH, Role.ADMIN, must_change_password=False)
    assert _code(exc) is ErrorCode.USERNAME_TAKEN
    with pytest.raises(PipelineError) as exc:
        users.create("no", HASH, Role.USER, must_change_password=False)
    assert _code(exc) is ErrorCode.INVALID_REQUEST
    assert users.get("0" * 32) is None and users.get_by_username("nobody") is None


def test_has_active_admin(users: SqliteUserStore) -> None:
    assert not users.has_active_admin()
    users.create("alice", HASH, Role.USER, must_change_password=False)
    assert not users.has_active_admin()
    admin = users.create("root", HASH, Role.ADMIN, must_change_password=False)
    assert users.has_active_admin()
    other = users.create("root2", HASH, Role.ADMIN, must_change_password=False)
    users.update(admin.id, is_active=False)
    assert users.has_active_admin()
    with pytest.raises(PipelineError):
        users.update(other.id, is_active=False)


def test_update_and_noop(users: SqliteUserStore) -> None:
    users.create("root", HASH, Role.ADMIN, must_change_password=False)
    u = users.create("alice", HASH, Role.USER, must_change_password=False)
    assert users.update(u.id, role=Role.USER, is_active=True) == u  # no-op, unchanged
    promoted = users.update(u.id, role=Role.ADMIN)
    assert promoted.role is Role.ADMIN and promoted.updated_at >= u.updated_at
    disabled = users.update(u.id, is_active=False)
    assert not disabled.is_active
    with pytest.raises(PipelineError) as exc:
        users.update("0" * 32, is_active=True)
    assert _code(exc) is ErrorCode.NOT_FOUND


@pytest.mark.parametrize("change", [{"role": Role.USER}, {"is_active": False}])
def test_last_admin_guard_on_update(users: SqliteUserStore, change: dict[str, object]) -> None:
    admin = users.create("root", HASH, Role.ADMIN, must_change_password=False)
    with pytest.raises(PipelineError) as exc:
        users.update(admin.id, **change)  # type: ignore[arg-type]
    assert _code(exc) is ErrorCode.LAST_ADMIN
    # a disabled admin doesn't count as the backup
    spare = users.create("spare", HASH, Role.ADMIN, must_change_password=False)
    users.update(spare.id, is_active=False)
    with pytest.raises(PipelineError):
        users.update(admin.id, **change)  # type: ignore[arg-type]
    users.update(spare.id, is_active=True)
    assert users.update(admin.id, **change).id == admin.id  # type: ignore[arg-type]


def test_last_admin_guard_on_delete(db_path: Path, users: SqliteUserStore) -> None:
    admin = users.create("root", HASH, Role.ADMIN, must_change_password=False)
    with pytest.raises(PipelineError) as exc:
        users.delete(admin.id)
    assert _code(exc) is ErrorCode.LAST_ADMIN
    users.create("root2", HASH, Role.ADMIN, must_change_password=False)
    assert users.delete(admin.id) == []
    assert users.get(admin.id) is None
    with pytest.raises(PipelineError) as exc:
        users.delete(admin.id)
    assert _code(exc) is ErrorCode.NOT_FOUND


def test_disabled_admin_can_be_deleted(users: SqliteUserStore) -> None:
    users.create("root", HASH, Role.ADMIN, must_change_password=False)
    other = users.create("root2", HASH, Role.ADMIN, must_change_password=False)
    users.update(other.id, is_active=False)
    users.delete(other.id)


def test_two_admins_demoting_each_other_concurrently(db_path: Path) -> None:
    store = SqliteUserStore(db_path)
    a = store.create("admin-a", HASH, Role.ADMIN, must_change_password=False)
    b = store.create("admin-b", HASH, Role.ADMIN, must_change_password=False)
    barrier = threading.Barrier(2)
    outcomes: list[str] = []
    lock = threading.Lock()

    def demote(target: str) -> None:
        own = SqliteUserStore(db_path)
        barrier.wait()
        try:
            own.update(target, role=Role.USER)
            result = "ok"
        except PipelineError as exc:
            result = exc.code.value
        with lock:
            outcomes.append(result)

    for _ in range(5):  # repeat to give the race a chance
        threads = [threading.Thread(target=demote, args=(t,)) for t in (a.id, b.id)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert sorted(outcomes) == ["last_admin", "ok"]
        assert store.has_active_admin()
        outcomes.clear()
        for uid in (a.id, b.id):  # restore both admins for the next round
            store.update(uid, role=Role.ADMIN)


def test_delete_returns_job_ids_and_removes_rows(db_path: Path, users: SqliteUserStore) -> None:
    users.create("root", HASH, Role.ADMIN, must_change_password=False)
    alice = users.create("alice", HASH, Role.USER, must_change_password=False)
    bob = users.create("bob", HASH, Role.USER, must_change_password=False)
    _job(db_path, "a" * 32, alice.id)
    _job(db_path, "b" * 32, alice.id)
    _job(db_path, "c" * 32, bob.id)
    sessions = SqliteSessionStore(db_path, idle_seconds=60, absolute_seconds=600)
    token = sessions.create(alice.id)
    assert sorted(users.delete(alice.id)) == ["a" * 32, "b" * 32]
    jobs = SqliteJobStore(db_path)
    assert jobs.get("a" * 32) is None and jobs.get("c" * 32) is not None
    assert sessions.resolve(token) is None
    with db.connection(db_path) as conn:
        assert conn.execute("SELECT count(*) FROM sessions").fetchone()[0] == 0


def test_list_with_job_counts(db_path: Path, users: SqliteUserStore) -> None:
    bob = users.create("bob", HASH, Role.USER, must_change_password=False)
    alice = users.create("alice", HASH, Role.USER, must_change_password=False)
    _job(db_path, "a" * 32, alice.id)
    _job(db_path, "b" * 32, alice.id)
    _job(db_path, "c" * 32, None)  # legacy: counted for nobody
    rows = users.list_with_job_counts()
    assert [(u.username, n) for u, n in rows] == [("alice", 2), ("bob", 0)]
    assert rows[1][0] == bob


def test_set_password_and_record_login(users: SqliteUserStore) -> None:
    u = users.create("alice", HASH, Role.USER, must_change_password=True)
    users.set_password(u.id, HASH.replace("$1$1$1$", "$2$1$1$"), must_change=False)
    after = users.get(u.id)
    assert after is not None and not after.must_change_password
    assert after.password_hash != u.password_hash
    assert after.password_changed_at >= u.password_changed_at
    users.record_login(u.id)
    logged = users.get(u.id)
    assert logged is not None and logged.last_login_at is not None
    with pytest.raises(PipelineError) as exc:
        users.set_password("0" * 32, HASH, must_change=False)
    assert _code(exc) is ErrorCode.NOT_FOUND


def test_create_admin_first_claims_orphans(db_path: Path, users: SqliteUserStore) -> None:
    _job(db_path, "a" * 32, None)
    _job(db_path, "b" * 32, None)
    admin, claimed = users.create_admin("root", HASH)
    assert claimed == 2 and admin.role is Role.ADMIN and not admin.must_change_password
    rec = SqliteJobStore(db_path).get("a" * 32)
    assert rec is not None and rec.owner_id == admin.id
    _job(db_path, "c" * 32, None)
    second, claimed2 = users.create_admin("root2", HASH)
    assert claimed2 is None
    rec = SqliteJobStore(db_path).get("c" * 32)
    assert rec is not None and rec.owner_id is None
    with pytest.raises(PipelineError) as exc:
        users.create_admin("root", HASH)
    assert _code(exc) is ErrorCode.USERNAME_TAKEN
    assert second.id != admin.id
