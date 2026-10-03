"""Job state store (PLAN §4.3): ``JobStore`` Protocol + stdlib-``sqlite3`` implementation.

One short-lived connection per operation (safe across the uvicorn loop and worker threads), WAL
journal so status polling never blocks progress writes. Progress throttling is the caller's job
(:class:`app.core.runner.ProgressReporter`); the store writes what it is given.

The schema lives in :mod:`app.core.db` (``init()`` runs its migrations). Every job has an
``owner_id`` (PLAN-auth U3); ``NULL`` marks a legacy job from before accounts (admin-only, A10).
Reads ``LEFT JOIN users`` to return the owner's username with the record.

``delete_jobs_of_owner`` / ``claim_orphan_jobs`` take an open connection so the user store can
run them inside its own ``BEGIN IMMEDIATE`` transaction (user deletion, first-admin bootstrap).
"""

from __future__ import annotations

import base64
import binascii
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, runtime_checkable

from app.api.schemas import JobOptions, JobSource
from app.core import db
from app.core.errors import ErrorCode, PipelineError
from app.core.stages import JobState, Stage

_COLUMNS = (
    "id, status, stage, progress, error_code, error_message, original_filename, ext, "
    "options_json, source_json, created_at, updated_at, owner_id"
)
#: ``SELECT`` list for reads: job columns + the owner's username (``LEFT JOIN users u``).
_SELECT = (
    "SELECT j.id, j.status, j.stage, j.progress, j.error_code, j.error_message, "
    "j.original_filename, j.ext, j.options_json, j.source_json, j.created_at, j.updated_at, "
    "j.owner_id, u.username AS owner_username "
    "FROM jobs j LEFT JOIN users u ON u.id = j.owner_id"
)

#: Bounds of ``list_jobs(limit=...)`` (the route validates the same range).
LIST_LIMIT_MAX = 200


def utcnow() -> datetime:
    """Timezone-aware UTC now (the store's only clock)."""
    return datetime.now(UTC)


def _now_iso() -> str:
    # Fixed microsecond precision keeps the stored text sortable (``created_at`` keyset order).
    return utcnow().isoformat(timespec="microseconds")


class InvalidCursor(ValueError):
    """``list_jobs(cursor=...)`` was not produced by this store (the route answers 422)."""


def encode_cursor(created_at: str, job_id: str) -> str:
    """Opaque keyset cursor: urlsafe-base64 of ``"<created_at>|<id>"`` (stored text)."""
    return base64.urlsafe_b64encode(f"{created_at}|{job_id}".encode()).decode("ascii")


def decode_cursor(cursor: str) -> tuple[str, str]:
    """``(created_at, id)`` from :func:`encode_cursor`. Raises :class:`InvalidCursor`."""
    try:
        raw = base64.urlsafe_b64decode(cursor.encode("ascii")).decode("utf-8")
    except (UnicodeError, ValueError, binascii.Error) as exc:
        raise InvalidCursor("malformed cursor") from exc
    created_at, sep, job_id = raw.partition("|")
    if not sep or not created_at or not job_id:
        raise InvalidCursor("malformed cursor")
    try:
        datetime.fromisoformat(created_at)
    except ValueError as exc:
        raise InvalidCursor("malformed cursor") from exc
    return created_at, job_id


def delete_jobs_of_owner(conn: sqlite3.Connection, owner_id: str) -> list[str]:
    """Delete every job row of ``owner_id`` on ``conn`` (caller's transaction). Returns the ids;
    the caller deletes their directories after commit."""
    ids = [r[0] for r in conn.execute("SELECT id FROM jobs WHERE owner_id = ?", (owner_id,))]
    conn.execute("DELETE FROM jobs WHERE owner_id = ?", (owner_id,))
    return ids


def claim_orphan_jobs(conn: sqlite3.Connection, owner_id: str) -> int:
    """Assign every legacy job (``owner_id IS NULL``) to ``owner_id`` on ``conn``. Returns the
    number of jobs claimed (A10: the first admin created via the CLI)."""
    cur = conn.execute("UPDATE jobs SET owner_id = ? WHERE owner_id IS NULL", (owner_id,))
    return cur.rowcount


@dataclass(frozen=True, slots=True)
class JobRecord:
    """One row of ``jobs``."""

    id: str
    status: JobState
    stage: Stage
    progress: float
    error_code: ErrorCode | None
    error_message: str | None
    original_filename: str
    ext: str
    options: JobOptions
    source: JobSource | None
    created_at: datetime
    updated_at: datetime
    #: Owning user id; None = legacy job from before accounts (admin-only).
    owner_id: str | None = None
    #: Owner's username (``LEFT JOIN users``); None for legacy jobs.
    owner_username: str | None = None

    @property
    def is_terminal(self) -> bool:
        return self.status.is_terminal


@dataclass(frozen=True, slots=True)
class JobPage:
    """One page of :meth:`JobStore.list_jobs`."""

    items: list[JobRecord]
    #: Pass back as ``cursor`` for the next page; None on the last page.
    next_cursor: str | None


@runtime_checkable
class JobStore(Protocol):
    """Persistent job state. Swappable later (D1)."""

    def init(self) -> None:
        """Create / migrate the schema (idempotent, ``app.core.db.migrate``)."""
        ...

    def create(
        self,
        job_id: str,
        *,
        original_filename: str,
        ext: str,
        options: JobOptions,
        source: JobSource | None = None,
        owner_id: str | None = None,
    ) -> JobRecord:
        """Insert a ``queued`` job owned by ``owner_id`` (None only for tools/tests).

        Raises ``PipelineError(unauthenticated)`` if ``owner_id`` names no user (deleted
        meanwhile); the check and the insert are one transaction."""
        ...

    def get(self, job_id: str) -> JobRecord | None: ...

    def update_progress(self, job_id: str, stage: Stage, progress: float) -> None:
        """Set stage + overall progress; moves a ``queued`` job to ``processing``."""
        ...

    def set_source(self, job_id: str, source: JobSource) -> None: ...

    def mark_succeeded(
        self,
        job_id: str,
        *,
        options: JobOptions | None = None,
        error: tuple[ErrorCode | str, str] | None = None,
    ) -> None:
        """``succeeded``, stage ``done``, progress 1. ``options`` replaces the stored options (a
        re-run with a new choice); ``error`` records a failed re-run whose previous result is
        kept (``JobStatus.error`` on a succeeded job). Otherwise the error is cleared."""
        ...

    def claim_rerun(self, job_id: str, options: JobOptions) -> bool:
        """Atomically ``succeeded`` → ``queued`` (stage ``queued``, progress 0, ``options`` =
        the re-run's choice) for an interpretation re-run. False if the job is not
        ``succeeded``."""
        ...

    def mark_failed(self, job_id: str, code: ErrorCode | str, message: str) -> None:
        """``failed`` with an error; the stage is kept so the UI can say where it failed."""
        ...

    def list_by_status(self, status: JobState) -> list[JobRecord]:
        """Jobs in a status, oldest first."""
        ...

    def list_jobs(
        self,
        *,
        owner_id: str | None = None,
        status: JobState | None = None,
        limit: int = 50,
        cursor: str | None = None,
    ) -> JobPage:
        """Newest first (``created_at`` desc, ``id`` desc), keyset-paginated.

        ``owner_id=None`` = every job incl. legacy ones (admin view); otherwise only that
        owner's jobs. ``limit`` in ``1..LIST_LIMIT_MAX``. Raises :class:`InvalidCursor` for a
        cursor not produced by this store and ``ValueError`` for a bad ``limit``.
        """
        ...

    def delete(self, job_id: str) -> bool:
        """Delete the job row. True if it existed. The caller deletes the directory after."""
        ...

    def delete_by_owner(self, owner_id: str) -> list[str]:
        """Delete every job row of ``owner_id``; returns the removed ids."""
        ...

    def claim_orphans(self, owner_id: str) -> int:
        """Assign every legacy (ownerless) job to ``owner_id``; returns the count."""
        ...


class SqliteJobStore:
    """:class:`JobStore` on a SQLite file (``data/mimic.db``)."""

    def __init__(self, db_path: Path, *, timeout_s: float = 10.0) -> None:
        self.db_path = Path(db_path)
        self._timeout_s = timeout_s

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Short-lived connection; commits on success, rolls back on error, always closes."""
        with db.connection(self.db_path, timeout_s=self._timeout_s) as conn:
            yield conn

    def init(self) -> None:
        db.migrate(self.db_path, timeout_s=self._timeout_s)

    # ---- writes -----------------------------------------------------------------------------

    def create(
        self,
        job_id: str,
        *,
        original_filename: str,
        ext: str,
        options: JobOptions,
        source: JobSource | None = None,
        owner_id: str | None = None,
    ) -> JobRecord:
        now = _now_iso()
        # BEGIN IMMEDIATE: the owner check and the insert can't interleave with a user delete
        # (``users.delete`` runs in the same kind of transaction). ``jobs.owner_id`` has no FK,
        # so without the check an upload validated while its owner was being deleted would leave
        # a job (and its files) behind with a dangling owner (PLAN-auth U2, A14).
        with db.transaction(self.db_path, timeout_s=self._timeout_s) as conn:
            if (
                owner_id is not None
                and not conn.execute("SELECT 1 FROM users WHERE id = ?", (owner_id,)).fetchone()
            ):
                raise PipelineError(ErrorCode.UNAUTHENTICATED)
            conn.execute(
                f"INSERT INTO jobs ({_COLUMNS}) "
                "VALUES (?, ?, ?, 0, NULL, NULL, ?, ?, ?, ?, ?, ?, ?)",
                (
                    job_id,
                    JobState.QUEUED.value,
                    Stage.QUEUED.value,
                    original_filename,
                    ext,
                    options.model_dump_json(),
                    source.model_dump_json() if source is not None else None,
                    now,
                    now,
                    owner_id,
                ),
            )
        record = self.get(job_id)
        assert record is not None
        return record

    def update_progress(self, job_id: str, stage: Stage, progress: float) -> None:
        progress = min(max(float(progress), 0.0), 1.0)
        with self._connect() as conn:
            conn.execute(
                "UPDATE jobs SET status = ?, stage = ?, progress = ?, updated_at = ? "
                "WHERE id = ? AND status IN (?, ?)",
                (
                    JobState.PROCESSING.value,
                    Stage(stage).value,
                    progress,
                    _now_iso(),
                    job_id,
                    JobState.QUEUED.value,
                    JobState.PROCESSING.value,
                ),
            )

    def set_source(self, job_id: str, source: JobSource) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE jobs SET source_json = ?, updated_at = ? WHERE id = ?",
                (source.model_dump_json(), _now_iso(), job_id),
            )

    def mark_succeeded(
        self,
        job_id: str,
        *,
        options: JobOptions | None = None,
        error: tuple[ErrorCode | str, str] | None = None,
    ) -> None:
        code, message = (ErrorCode(error[0]).value, error[1]) if error else (None, None)
        with self._connect() as conn:
            conn.execute(
                "UPDATE jobs SET status = ?, stage = ?, progress = 1.0, error_code = ?, "
                "error_message = ?, options_json = COALESCE(?, options_json), updated_at = ? "
                "WHERE id = ?",
                (
                    JobState.SUCCEEDED.value,
                    Stage.DONE.value,
                    code,
                    message,
                    options.model_dump_json() if options is not None else None,
                    _now_iso(),
                    job_id,
                ),
            )

    def claim_rerun(self, job_id: str, options: JobOptions) -> bool:
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE jobs SET status = ?, stage = ?, progress = 0, error_code = NULL, "
                "error_message = NULL, options_json = ?, updated_at = ? "
                "WHERE id = ? AND status = ?",
                (
                    JobState.QUEUED.value,
                    Stage.QUEUED.value,
                    options.model_dump_json(),
                    _now_iso(),
                    job_id,
                    JobState.SUCCEEDED.value,
                ),
            )
            return cur.rowcount == 1

    def mark_failed(self, job_id: str, code: ErrorCode | str, message: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE jobs SET status = ?, error_code = ?, error_message = ?, updated_at = ? "
                "WHERE id = ?",
                (
                    JobState.FAILED.value,
                    ErrorCode(code).value,
                    message,
                    _now_iso(),
                    job_id,
                ),
            )

    def delete(self, job_id: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
            return cur.rowcount == 1

    def delete_by_owner(self, owner_id: str) -> list[str]:
        with db.transaction(self.db_path, timeout_s=self._timeout_s) as conn:
            return delete_jobs_of_owner(conn, owner_id)

    def claim_orphans(self, owner_id: str) -> int:
        with db.transaction(self.db_path, timeout_s=self._timeout_s) as conn:
            return claim_orphan_jobs(conn, owner_id)

    # ---- reads ------------------------------------------------------------------------------

    def get(self, job_id: str) -> JobRecord | None:
        with self._connect() as conn:
            row = conn.execute(f"{_SELECT} WHERE j.id = ?", (job_id,)).fetchone()
        return _to_record(row) if row is not None else None

    def list_by_status(self, status: JobState) -> list[JobRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                f"{_SELECT} WHERE j.status = ? ORDER BY j.created_at, j.id",
                (JobState(status).value,),
            ).fetchall()
        return [_to_record(r) for r in rows]

    def list_jobs(
        self,
        *,
        owner_id: str | None = None,
        status: JobState | None = None,
        limit: int = 50,
        cursor: str | None = None,
    ) -> JobPage:
        if not 1 <= limit <= LIST_LIMIT_MAX:
            raise ValueError(f"limit must be in 1..{LIST_LIMIT_MAX}")
        where: list[str] = []
        params: list[object] = []
        if owner_id is not None:
            where.append("j.owner_id = ?")
            params.append(owner_id)
        if status is not None:
            where.append("j.status = ?")
            params.append(JobState(status).value)
        if cursor is not None:
            created_at, job_id = decode_cursor(cursor)
            where.append("(j.created_at, j.id) < (?, ?)")
            params += [created_at, job_id]
        sql = _SELECT
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY j.created_at DESC, j.id DESC LIMIT ?"
        params.append(limit + 1)
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        next_cursor = None
        if len(rows) > limit:
            last = rows[limit - 1]
            next_cursor = encode_cursor(last["created_at"], last["id"])
            rows = rows[:limit]
        return JobPage(items=[_to_record(r) for r in rows], next_cursor=next_cursor)


def _to_record(row: sqlite3.Row) -> JobRecord:
    return JobRecord(
        id=row["id"],
        status=JobState(row["status"]),
        stage=Stage(row["stage"]),
        progress=float(row["progress"]),
        error_code=ErrorCode(row["error_code"]) if row["error_code"] else None,
        error_message=row["error_message"],
        original_filename=row["original_filename"],
        ext=row["ext"],
        options=JobOptions.model_validate_json(row["options_json"]),
        source=(JobSource.model_validate_json(row["source_json"]) if row["source_json"] else None),
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
        owner_id=row["owner_id"],
        owner_username=row["owner_username"],
    )
