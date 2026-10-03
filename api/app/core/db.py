"""SQLite connection helpers + schema migrations for ``data/mimic.db`` (PLAN-auth §2, A15).

Schema versioning uses ``PRAGMA user_version``. :func:`migrate` applies ordered, idempotent
steps inside one ``BEGIN IMMEDIATE`` transaction (SQLite DDL is transactional, so a failed
migration leaves the previous version intact). Steps use individual ``execute`` calls, never
``executescript`` (which implicitly COMMITs).

Before the v0 → v1 migration of a database that already holds jobs, an online backup is written
next to it: ``mimic.db.pre-auth-<YYYYmmddTHHMMSSZ>.bak``.

Every connection: ``timeout`` 10 s, ``row_factory = sqlite3.Row``, ``PRAGMA foreign_keys=ON``
(needed for the ``sessions`` → ``users`` cascade). WAL is set once, in :func:`migrate`.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

log = logging.getLogger(__name__)

#: Schema version this code writes and understands.
SCHEMA_VERSION = 1

DEFAULT_TIMEOUT_S = 10.0


# ---- connections ----------------------------------------------------------------------------


def connect(
    db_path: Path | str, *, timeout_s: float = DEFAULT_TIMEOUT_S, autocommit: bool = False
) -> sqlite3.Connection:
    """New configured connection; the caller closes it.

    ``autocommit=True`` sets ``isolation_level=None`` so the caller can issue an explicit
    ``BEGIN IMMEDIATE`` (see :func:`transaction`). Otherwise the stdlib's implicit transactions
    apply (use ``with conn:`` to commit / roll back).
    """
    conn = sqlite3.connect(db_path, timeout=timeout_s)
    if autocommit:
        conn.isolation_level = None
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def connection(
    db_path: Path | str, *, timeout_s: float = DEFAULT_TIMEOUT_S
) -> Iterator[sqlite3.Connection]:
    """Short-lived connection: commits on success, rolls back on error, always closes."""
    with closing(connect(db_path, timeout_s=timeout_s)) as conn, conn:
        yield conn


@contextmanager
def transaction(
    db_path: Path | str, *, timeout_s: float = DEFAULT_TIMEOUT_S
) -> Iterator[sqlite3.Connection]:
    """``BEGIN IMMEDIATE`` … ``COMMIT`` (``ROLLBACK`` + re-raise on error), always closes.

    Takes the write lock up front, so read-check-write sequences (last-admin guard, orphan
    claim) cannot interleave with another writer.
    """
    with closing(connect(db_path, timeout_s=timeout_s, autocommit=True)) as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        conn.execute("COMMIT")


# ---- migrations -----------------------------------------------------------------------------


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)
    ).fetchone()
    return row is not None


def _to_v1(conn: sqlite3.Connection) -> None:
    """v0 (MVP job store) → v1: ``jobs.owner_id`` + ``users`` + ``sessions``."""
    # jobs: unchanged MVP columns (created here for a fresh database)
    conn.execute(
        """
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
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS jobs_status_created ON jobs (status, created_at)")
    if "owner_id" not in _columns(conn, "jobs"):
        # NULL = legacy job, admin-only (A10). No FK: user deletion removes jobs explicitly.
        conn.execute("ALTER TABLE jobs ADD COLUMN owner_id TEXT")
    conn.execute("CREATE INDEX IF NOT EXISTS jobs_owner_created ON jobs (owner_id, created_at, id)")
    conn.execute("CREATE INDEX IF NOT EXISTS jobs_created ON jobs (created_at, id)")

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id                   TEXT PRIMARY KEY,
            username             TEXT NOT NULL UNIQUE CHECK (username = lower(username)),
            password_hash        TEXT NOT NULL,
            role                 TEXT NOT NULL CHECK (role IN ('admin', 'user')),
            is_active            INTEGER NOT NULL DEFAULT 1 CHECK (is_active IN (0, 1)),
            must_change_password INTEGER NOT NULL DEFAULT 0
                                 CHECK (must_change_password IN (0, 1)),
            created_at           TEXT NOT NULL,
            updated_at           TEXT NOT NULL,
            password_changed_at  TEXT NOT NULL,
            last_login_at        TEXT,
            created_by           TEXT
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS users_role_active ON users (role, is_active)")

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            token_hash   TEXT PRIMARY KEY,
            user_id      TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            created_at   TEXT NOT NULL,
            last_seen_at TEXT NOT NULL,
            expires_at   TEXT NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS sessions_user ON sessions (user_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS sessions_expires ON sessions (expires_at)")


#: ``MIGRATIONS[i]`` upgrades version ``i`` to ``i + 1``. Append only; never edit a shipped step.
MIGRATIONS: tuple[Callable[[sqlite3.Connection], None], ...] = (_to_v1,)
assert len(MIGRATIONS) == SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class MigrationResult:
    from_version: int
    to_version: int
    backup_path: Path | None

    @property
    def migrated(self) -> bool:
        return self.to_version != self.from_version


def user_version(conn: sqlite3.Connection) -> int:
    return int(conn.execute("PRAGMA user_version").fetchone()[0])


def _check_not_newer(version: int) -> None:
    if version > SCHEMA_VERSION:
        raise RuntimeError(
            f"database schema v{version} is newer than this code (v{SCHEMA_VERSION}); upgrade mimic"
        )


def _backup_path(db_path: Path, now: datetime) -> Path:
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    candidate = db_path.with_name(f"{db_path.name}.pre-auth-{stamp}.bak")
    n = 1
    while candidate.exists():
        candidate = db_path.with_name(f"{db_path.name}.pre-auth-{stamp}-{n}.bak")
        n += 1
    return candidate


def _backup(conn: sqlite3.Connection, db_path: Path) -> Path:
    """Online backup (consistent snapshot incl. WAL content) as a self-contained file."""
    dst_path = _backup_path(db_path, datetime.now(UTC))
    with closing(sqlite3.connect(dst_path)) as dst:
        conn.backup(dst)
        dst.execute("PRAGMA journal_mode=DELETE")
    return dst_path


def migrate(db_path: Path | str, *, timeout_s: float = DEFAULT_TIMEOUT_S) -> MigrationResult:
    """Bring the database at ``db_path`` to :data:`SCHEMA_VERSION` (idempotent).

    Raises ``RuntimeError`` if the database is newer than this code. Safe to run while another
    process uses the database (write lock + version re-check inside the transaction).
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with closing(connect(db_path, timeout_s=timeout_s, autocommit=True)) as conn:
        start = user_version(conn)
        _check_not_newer(start)
        backup: Path | None = None
        if (
            start == 0
            and _table_exists(conn, "jobs")
            and conn.execute("SELECT count(*) FROM jobs").fetchone()[0] > 0
        ):
            backup = _backup(conn, db_path)
            log.info("db: backed up schema v0 database to %s before migrating", backup)
        conn.execute("PRAGMA journal_mode=WAL")

        conn.execute("BEGIN IMMEDIATE")
        try:
            current = user_version(conn)  # another process may have migrated meanwhile
            _check_not_newer(current)
            for step in MIGRATIONS[current:]:
                step(conn)
            if current != SCHEMA_VERSION:
                conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        conn.execute("COMMIT")

    if current != SCHEMA_VERSION:
        log.info("db: migrated %s from schema v%d to v%d", db_path, current, SCHEMA_VERSION)
    return MigrationResult(from_version=current, to_version=SCHEMA_VERSION, backup_path=backup)
