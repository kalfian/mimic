"""SqliteJobStore + ProgressReporter (Track A)."""

from __future__ import annotations

import sqlite3
import threading
import time
import uuid
from pathlib import Path

import pytest

from app.api.schemas import JobOptions, JobSource
from app.core.errors import ErrorCode
from app.core.jobstore import JobStore, SqliteJobStore
from app.core.runner import JobInterrupted, ProgressReporter
from app.core.stages import JobState, Stage, overall_progress


@pytest.fixture
def store(tmp_path: Path) -> SqliteJobStore:
    s = SqliteJobStore(tmp_path / "mimic.db")
    s.init()
    return s


def _new(store: SqliteJobStore, **kw: object) -> str:
    job_id = uuid.uuid4().hex
    store.create(
        job_id,
        original_filename=str(kw.get("filename", "clip.mp4")),
        ext="mp4",
        options=JobOptions(pixel_ratio="2", use_interpreter=True),
        source=kw.get("source"),  # type: ignore[arg-type]
    )
    return job_id


def test_implements_protocol(store: SqliteJobStore) -> None:
    assert isinstance(store, JobStore)


def test_init_is_idempotent_and_uses_wal(store: SqliteJobStore) -> None:
    store.init()
    with sqlite3.connect(store.db_path) as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


def test_create_and_get(store: SqliteJobStore) -> None:
    src = JobSource(filename="clip.mp4", width=1280, height=720, fps=60.0, duration_s=3.2)
    job_id = _new(store, source=src)
    rec = store.get(job_id)
    assert rec is not None
    assert rec.status is JobState.QUEUED and rec.stage is Stage.QUEUED
    assert rec.progress == 0.0
    assert rec.error_code is None and rec.error_message is None
    assert rec.options == JobOptions(pixel_ratio="2", use_interpreter=True)
    assert rec.source == src
    assert rec.created_at.tzinfo is not None
    assert rec.created_at == rec.updated_at
    assert store.get(uuid.uuid4().hex) is None


def test_progress_moves_to_processing(store: SqliteJobStore) -> None:
    job_id = _new(store)
    store.update_progress(job_id, Stage.SCANNING, 0.2)
    rec = store.get(job_id)
    assert rec is not None
    assert (rec.status, rec.stage, rec.progress) == (JobState.PROCESSING, Stage.SCANNING, 0.2)
    assert rec.updated_at >= rec.created_at
    store.update_progress(job_id, Stage.SCANNING, 7.0)
    assert store.get(job_id).progress == 1.0  # type: ignore[union-attr]


def test_succeeded(store: SqliteJobStore) -> None:
    job_id = _new(store)
    store.update_progress(job_id, Stage.GENERATING, 0.97)
    store.mark_succeeded(job_id)
    rec = store.get(job_id)
    assert rec is not None and rec.is_terminal
    assert (rec.status, rec.stage, rec.progress) == (JobState.SUCCEEDED, Stage.DONE, 1.0)


def test_failed_keeps_stage(store: SqliteJobStore) -> None:
    job_id = _new(store)
    store.update_progress(job_id, Stage.SCANNING, 0.15)
    store.mark_failed(job_id, ErrorCode.NO_MOTION_DETECTED, "nothing moved")
    rec = store.get(job_id)
    assert rec is not None
    assert rec.status is JobState.FAILED and rec.stage is Stage.SCANNING
    assert (rec.error_code, rec.error_message) == (ErrorCode.NO_MOTION_DETECTED, "nothing moved")


def test_terminal_jobs_ignore_progress(store: SqliteJobStore) -> None:
    job_id = _new(store)
    store.mark_failed(job_id, "interrupted", "restart")
    store.update_progress(job_id, Stage.MEASURING, 0.6)
    rec = store.get(job_id)
    assert rec is not None and rec.status is JobState.FAILED and rec.stage is Stage.QUEUED


def test_list_by_status_oldest_first(store: SqliteJobStore) -> None:
    ids = []
    for _ in range(3):
        ids.append(_new(store))
        time.sleep(0.002)
    store.update_progress(ids[1], Stage.PROBING, 0.0)
    assert [r.id for r in store.list_by_status(JobState.QUEUED)] == [ids[0], ids[2]]
    assert [r.id for r in store.list_by_status(JobState.PROCESSING)] == [ids[1]]
    assert store.list_by_status(JobState.SUCCEEDED) == []


def test_set_source_and_persistence(store: SqliteJobStore) -> None:
    job_id = _new(store)
    src = JobSource(filename="a.webm", width=10, height=20, fps=30.0, duration_s=1.0)
    store.set_source(job_id, src)
    reopened = SqliteJobStore(store.db_path)
    rec = reopened.get(job_id)
    assert rec is not None and rec.source == src


def test_concurrent_writes(store: SqliteJobStore) -> None:
    ids = [_new(store) for _ in range(4)]

    def work(job_id: str) -> None:
        for i in range(25):
            store.update_progress(job_id, Stage.MEASURING, i / 25)

    threads = [threading.Thread(target=work, args=(j,)) for j in ids]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert all(store.get(j).progress == pytest.approx(24 / 25) for j in ids)  # type: ignore[union-attr]


# ---- ProgressReporter ----------------------------------------------------------------------


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple[Stage, float]] = []

    def update_progress(self, job_id: str, stage: Stage, progress: float) -> None:
        self.calls.append((stage, progress))


def test_reporter_maps_windows_and_throttles() -> None:
    rec = _Recorder()
    r = ProgressReporter(rec, "x", throttle_s=10.0)  # type: ignore[arg-type]
    r.enter(Stage.SCANNING)
    for f in (0.1, 0.5, 0.9):
        r.update(f)
    assert rec.calls == [(Stage.SCANNING, overall_progress(Stage.SCANNING, 0.0))]
    assert r.progress == pytest.approx(overall_progress(Stage.SCANNING, 0.9))
    r.enter(Stage.DECODING)  # stage change is always written
    assert rec.calls[-1] == (Stage.DECODING, pytest.approx(0.30))


def test_reporter_is_monotonic() -> None:
    rec = _Recorder()
    r = ProgressReporter(rec, "x", throttle_s=0.0)  # type: ignore[arg-type]
    r.enter(Stage.MEASURING, 0.8)
    r.update(0.1)
    assert [p for _, p in rec.calls] == [pytest.approx(overall_progress(Stage.MEASURING, 0.8))] * 2


def test_reporter_interrupts_on_stop() -> None:
    stop = threading.Event()
    r = ProgressReporter(_Recorder(), "x", stop)  # type: ignore[arg-type]
    r.enter(Stage.PROBING)
    stop.set()
    assert r.stopping
    for call in (lambda: r.update(0.5), r.checkpoint, lambda: r.enter(Stage.PREVIEW)):
        with pytest.raises(JobInterrupted):
            call()
    with pytest.raises(JobInterrupted):
        r.sleep(5.0)  # returns immediately when stopping


def test_reporter_creeping_advances_progress() -> None:
    rec = _Recorder()
    r = ProgressReporter(rec, "x", throttle_s=0.0)  # type: ignore[arg-type]
    r.enter(Stage.INTERPRETING)
    with r.creeping(expected_s=0.05, tick_s=0.01):
        time.sleep(0.15)
    lo, hi = overall_progress(Stage.INTERPRETING, 0.0), overall_progress(Stage.INTERPRETING, 0.95)
    assert lo < r.progress <= hi + 1e-9
    assert len(rec.calls) > 2


def test_reporter_swallows_store_errors() -> None:
    class Broken:
        def update_progress(self, *_: object) -> None:
            raise sqlite3.OperationalError("database is locked")

    r = ProgressReporter(Broken(), "x")  # type: ignore[arg-type]
    r.enter(Stage.PROBING)
    r.update(1.0)


def test_claim_rerun_only_from_succeeded(store: SqliteJobStore) -> None:
    job_id = _new(store)
    new_opts = JobOptions(pixel_ratio="2", use_interpreter=False)
    assert store.claim_rerun(job_id, new_opts) is False  # queued
    store.update_progress(job_id, Stage.MEASURING, 0.5)
    assert store.claim_rerun(job_id, new_opts) is False  # processing
    store.mark_succeeded(job_id)
    assert store.claim_rerun(job_id, new_opts) is True
    rec = store.get(job_id)
    assert rec is not None
    assert (rec.status, rec.stage, rec.progress) == (JobState.QUEUED, Stage.QUEUED, 0.0)
    assert rec.options == new_opts
    assert store.claim_rerun(job_id, new_opts) is False  # second click


def test_mark_succeeded_with_options_and_rerun_error(store: SqliteJobStore) -> None:
    job_id = _new(store)
    store.mark_succeeded(job_id, options=JobOptions(use_interpreter=False),
                         error=(ErrorCode.INTERNAL_ERROR, "re-run failed"))  # fmt: skip
    rec = store.get(job_id)
    assert rec is not None and rec.status is JobState.SUCCEEDED
    assert (rec.error_code, rec.error_message) == (ErrorCode.INTERNAL_ERROR, "re-run failed")
    assert rec.options.use_interpreter is False and rec.options.pixel_ratio == "auto"
    store.mark_succeeded(job_id)  # a later success clears the error, keeps the options
    rec = store.get(job_id)
    assert rec is not None and rec.error_code is None and rec.options.use_interpreter is False
