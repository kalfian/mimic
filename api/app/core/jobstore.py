"""Job state store (PLAN §4.3): ``JobStore`` Protocol + stdlib-``sqlite3`` implementation.

One short-lived connection per operation (safe across the uvicorn loop and worker threads), WAL
journal so status polling never blocks progress writes. Progress throttling is the caller's job
(:class:`app.core.runner.ProgressReporter`); the store writes what it is given.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, runtime_checkable

from app.api.schemas import JobOptions, JobSource
from app.core.errors import ErrorCode
from app.core.stages import JobState, Stage

_SCHEMA = """
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

_COLUMNS = (
    "id, status, stage, progress, error_code, error_message, original_filename, ext, "
    "options_json, source_json, created_at, updated_at"
)


def utcnow() -> datetime:
    """Timezone-aware UTC now (the store's only clock)."""
    return datetime.now(UTC)


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

    @property
    def is_terminal(self) -> bool:
        return self.status.is_terminal


@runtime_checkable
class JobStore(Protocol):
    """Persistent job state. Swappable later (D1)."""

    def init(self) -> None:
        """Create the schema (idempotent)."""
        ...

    def create(
        self,
        job_id: str,
        *,
        original_filename: str,
        ext: str,
        options: JobOptions,
        source: JobSource | None = None,
    ) -> JobRecord:
        """Insert a ``queued`` job."""
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


class SqliteJobStore:
    """:class:`JobStore` on a SQLite file (``data/mimic.db``)."""

    def __init__(self, db_path: Path, *, timeout_s: float = 10.0) -> None:
        self.db_path = Path(db_path)
        self._timeout_s = timeout_s

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Short-lived connection; commits on success, rolls back on error, always closes."""
        with closing(sqlite3.connect(self.db_path, timeout=self._timeout_s)) as conn:
            conn.row_factory = sqlite3.Row
            with conn:
                yield conn

    def init(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(_SCHEMA)

    # ---- writes -----------------------------------------------------------------------------

    def create(
        self,
        job_id: str,
        *,
        original_filename: str,
        ext: str,
        options: JobOptions,
        source: JobSource | None = None,
    ) -> JobRecord:
        now = utcnow().isoformat()
        with self._connect() as conn:
            conn.execute(
                f"INSERT INTO jobs ({_COLUMNS}) VALUES (?, ?, ?, 0, NULL, NULL, ?, ?, ?, ?, ?, ?)",
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
                    utcnow().isoformat(),
                    job_id,
                    JobState.QUEUED.value,
                    JobState.PROCESSING.value,
                ),
            )

    def set_source(self, job_id: str, source: JobSource) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE jobs SET source_json = ?, updated_at = ? WHERE id = ?",
                (source.model_dump_json(), utcnow().isoformat(), job_id),
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
                    utcnow().isoformat(),
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
                    utcnow().isoformat(),
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
                    utcnow().isoformat(),
                    job_id,
                ),
            )

    # ---- reads ------------------------------------------------------------------------------

    def get(self, job_id: str) -> JobRecord | None:
        with self._connect() as conn:
            row = conn.execute(f"SELECT {_COLUMNS} FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return _to_record(row) if row is not None else None

    def list_by_status(self, status: JobState) -> list[JobRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT {_COLUMNS} FROM jobs WHERE status = ? ORDER BY created_at, id",
                (JobState(status).value,),
            ).fetchall()
        return [_to_record(r) for r in rows]


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
    )
