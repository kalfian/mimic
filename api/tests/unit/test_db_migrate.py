"""``app.core.db``: connections, schema v1 migration, pre-auth backup (PLAN-auth §2.1, A15)."""

from __future__ import annotations

import sqlite3
import threading
import uuid
from contextlib import closing
from pathlib import Path

import pytest

from app.api.schemas import JobOptions
from app.core import db
from app.core.jobstore import SqliteJobStore
from app.core.stages import JobState
from tests.auth_helpers import insert_user

#: The MVP (schema v0) job store DDL, verbatim from before accounts existed.
V0_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id                TEXT PRIMARY KEY,
    status            TEXT NOT NULL,
    stage             TEXT NOT NULL,
    progress          REAL NOT NULL DEFAULT 0,
    error_code        TEXT,
    error_message     TEXT,
    original_filename TEXT NOT NULL,
    ext               TEXT NOT NULL,
    options_json      TEXT NOT NULL,
    source_json       TEXT,
    created_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS jobs_status_created ON jobs (status, created_at);
"""

V1_INDEXES = {
    "jobs_status_created",
    "jobs_owner_created",
    "jobs_created",
    "users_role_active",
    "sessions_user",
    "sessions_expires",
}


def make_v0_db(path: Path, rows: int) -> list[str]:
    """A v0 database like the MVP wrote it (WAL, ``user_version`` 0) with ``rows`` jobs."""
    ids = [uuid.uuid4().hex for _ in range(rows)]
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(V0_SCHEMA)
        for i, job_id in enumerate(ids):
            conn.execute(
                "INSERT INTO jobs (id, status, stage, progress, error_code, error_message, "
                "original_filename, ext, options_json, source_json, created_at, updated_at) "
                "VALUES (?, 'succeeded', 'done', 1.0, NULL, NULL, ?, 'mp4', ?, NULL, ?, ?)",
                (
                    job_id,
                    f"clip{i}.mp4",
                    JobOptions().model_dump_json(),
                    f"2026-10-0{i % 9 + 1}T10:00:00+00:00",
                    f"2026-10-0{i % 9 + 1}T10:00:05+00:00",
                ),
            )
        conn.commit()
    return ids


def version(path: Path) -> int:
    with closing(sqlite3.connect(path)) as conn:
        return db.user_version(conn)


def tables(path: Path) -> set[str]:
    with closing(sqlite3.connect(path)) as conn:
        return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def indexes(path: Path) -> set[str]:
    with closing(sqlite3.connect(path)) as conn:
        return {
            r[0]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='index'")
            if not r[0].startswith("sqlite_autoindex")
        }


def job_columns(path: Path) -> set[str]:
    with closing(sqlite3.connect(path)) as conn:
        return {r[1] for r in conn.execute("PRAGMA table_info(jobs)")}


def backups(path: Path) -> list[Path]:
    return sorted(path.parent.glob(f"{path.name}.pre-auth-*.bak"))


# ---- fresh database --------------------------------------------------------------------------


def test_fresh_db_gets_v1_schema(tmp_path: Path) -> None:
    path = tmp_path / "data" / "mimic.db"  # parent dir is created
    result = db.migrate(path)
    assert (result.from_version, result.to_version, result.backup_path) == (0, 1, None)
    assert result.migrated
    assert version(path) == db.SCHEMA_VERSION == 1
    assert {"jobs", "users", "sessions"} <= tables(path)
    assert V1_INDEXES <= indexes(path)
    assert "owner_id" in job_columns(path)
    with closing(sqlite3.connect(path)) as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert backups(path) == []


def test_second_run_is_noop(tmp_path: Path) -> None:
    path = tmp_path / "mimic.db"
    db.migrate(path)
    again = db.migrate(path)
    assert (again.from_version, again.to_version, again.migrated) == (1, 1, False)
    assert again.backup_path is None


# ---- existing v0 database ---------------------------------------------------------------------


def test_v0_with_rows_is_backed_up_and_migrated(tmp_path: Path) -> None:
    path = tmp_path / "mimic.db"
    ids = make_v0_db(path, rows=3)
    assert version(path) == 0

    result = db.migrate(path)
    assert (result.from_version, result.to_version) == (0, 1)
    assert result.backup_path is not None and result.backup_path.exists()
    assert backups(path) == [result.backup_path]
    assert version(path) == 1
    assert V1_INDEXES <= indexes(path)

    # every row is intact and readable through the store; legacy jobs have no owner
    store = SqliteJobStore(path)
    for i, job_id in enumerate(ids):
        rec = store.get(job_id)
        assert rec is not None
        assert rec.original_filename == f"clip{i}.mp4"
        assert rec.status is JobState.SUCCEEDED
        assert (rec.owner_id, rec.owner_username) == (None, None)

    # the backup is a self-contained v0 snapshot with the same rows
    bak = result.backup_path
    with closing(sqlite3.connect(bak)) as conn:
        assert db.user_version(conn) == 0
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
        assert sorted(r[0] for r in conn.execute("SELECT id FROM jobs")) == sorted(ids)
    assert "owner_id" not in job_columns(bak)
    assert not Path(f"{bak}-wal").exists()

    # second run: nothing to do, no second backup
    assert db.migrate(path).migrated is False
    assert backups(path) == [bak]


def test_v0_without_rows_is_not_backed_up(tmp_path: Path) -> None:
    path = tmp_path / "mimic.db"
    make_v0_db(path, rows=0)
    result = db.migrate(path)
    assert result.backup_path is None and backups(path) == []
    assert version(path) == 1 and "owner_id" in job_columns(path)


def test_backup_names_do_not_collide(tmp_path: Path) -> None:
    path = tmp_path / "mimic.db"
    make_v0_db(path, rows=1)
    first = db.migrate(path).backup_path
    # simulate a v0 database again within the same second
    path.unlink()
    for suffix in ("-wal", "-shm"):
        Path(f"{path}{suffix}").unlink(missing_ok=True)
    make_v0_db(path, rows=1)
    second = db.migrate(path).backup_path
    assert first is not None and second is not None and first != second
    assert first.exists() and second.exists()


def test_v0_migration_runs_through_store_init(tmp_path: Path) -> None:
    path = tmp_path / "mimic.db"
    ids = make_v0_db(path, rows=2)
    store = SqliteJobStore(path)
    store.init()
    assert version(path) == 1 and len(backups(path)) == 1
    assert {r.id for r in store.list_by_status(JobState.SUCCEEDED)} == set(ids)


# ---- refusal / rollback ----------------------------------------------------------------------


def test_newer_schema_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "mimic.db"
    db.migrate(path)
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("PRAGMA user_version = 7")
    with pytest.raises(RuntimeError, match=r"schema v7 is newer than this code \(v1\)"):
        db.migrate(path)
    assert version(path) == 7


def test_failed_migration_rolls_back(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "mimic.db"
    ids = make_v0_db(path, rows=2)

    def broken(conn: sqlite3.Connection) -> None:
        db._to_v1(conn)  # every DDL statement succeeds ...
        raise sqlite3.OperationalError("disk I/O error (simulated)")  # ... then the step fails

    monkeypatch.setattr(db, "MIGRATIONS", (broken,))
    with pytest.raises(sqlite3.OperationalError, match="simulated"):
        db.migrate(path)

    assert version(path) == 0
    assert tables(path) == {"jobs"}
    assert "owner_id" not in job_columns(path)
    assert "jobs_owner_created" not in indexes(path)
    with closing(sqlite3.connect(path)) as conn:
        assert sorted(r[0] for r in conn.execute("SELECT id FROM jobs")) == sorted(ids)
    assert len(backups(path)) == 1  # written before the transaction

    monkeypatch.undo()
    assert db.migrate(path).to_version == 1 and "owner_id" in job_columns(path)


def test_concurrent_migrations(tmp_path: Path) -> None:
    path = tmp_path / "mimic.db"
    make_v0_db(path, rows=1)
    errors: list[BaseException] = []

    def run() -> None:
        try:
            db.migrate(path)
        except BaseException as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=run) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert version(path) == 1 and "owner_id" in job_columns(path)


# ---- connections -----------------------------------------------------------------------------


def test_connect_enables_foreign_keys_and_session_cascade(tmp_path: Path) -> None:
    path = tmp_path / "mimic.db"
    user_id = insert_user(path, "alice")
    with db.connection(path) as conn:
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        conn.execute(
            "INSERT INTO sessions (token_hash, user_id, created_at, last_seen_at, expires_at) "
            "VALUES ('h', ?, 'x', 'x', 'x')",
            (user_id,),
        )
    with db.connection(path) as conn:
        conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
    with db.connection(path) as conn:
        assert conn.execute("SELECT count(*) FROM sessions").fetchone()[0] == 0
        with pytest.raises(sqlite3.IntegrityError):  # FK enforced on insert too
            conn.execute(
                "INSERT INTO sessions (token_hash, user_id, created_at, last_seen_at, "
                "expires_at) VALUES ('h2', 'nobody', 'x', 'x', 'x')"
            )


def test_users_table_constraints(tmp_path: Path) -> None:
    path = tmp_path / "mimic.db"
    insert_user(path, "alice")
    with pytest.raises(sqlite3.IntegrityError):  # unique
        insert_user(path, "alice")
    with pytest.raises(sqlite3.IntegrityError):  # lowercase only
        insert_user(path, "Bob")


def test_users_role_check_rejects_unknown_role(tmp_path: Path) -> None:
    path = tmp_path / "mimic.db"
    db.migrate(path)
    with db.connection(path) as conn, pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO users (id, username, password_hash, role, created_at, updated_at, "
            "password_changed_at) VALUES ('1', 'dave', '!', 'root', 'x', 'x', 'x')"
        )


def test_transaction_commits_and_rolls_back(tmp_path: Path) -> None:
    path = tmp_path / "mimic.db"
    insert_user(path, "alice")
    with pytest.raises(RuntimeError), db.transaction(path) as conn:
        conn.execute("DELETE FROM users")
        raise RuntimeError("abort")
    with db.connection(path) as conn:
        assert conn.execute("SELECT count(*) FROM users").fetchone()[0] == 1
    with db.transaction(path) as conn:
        conn.execute("DELETE FROM users")
    with db.connection(path) as conn:
        assert conn.execute("SELECT count(*) FROM users").fetchone()[0] == 0
