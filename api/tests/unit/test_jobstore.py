"""SqliteJobStore + ProgressReporter (Track A)."""

from __future__ import annotations

import sqlite3
import threading
import time
import uuid
from pathlib import Path

import pytest

from app.api.schemas import JobOptions, JobSource
from app.core import db
from app.core.errors import ErrorCode
from app.core.jobstore import (
    InvalidCursor,
    JobStore,
    SqliteJobStore,
    decode_cursor,
    encode_cursor,
)
from app.core.runner import JobInterrupted, ProgressReporter
from app.core.stages import JobState, Stage, overall_progress
from tests.auth_helpers import insert_user


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
        owner_id=kw.get("owner_id"),  # type: ignore[arg-type]
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


# ---- ownership, listing, deletion (PLAN-auth §2, §3.3, §14) ---------------------------------


def _set_created(store: SqliteJobStore, job_id: str, created_at: str) -> None:
    with db.connection(store.db_path) as conn:
        conn.execute("UPDATE jobs SET created_at = ? WHERE id = ?", (created_at, job_id))


def _page_ids(store: SqliteJobStore, **kw: object) -> list[str]:
    return [r.id for r in store.list_jobs(**kw).items]  # type: ignore[arg-type]


def test_owner_on_create_and_get(store: SqliteJobStore) -> None:
    alice = insert_user(store.db_path, "alice")
    owned = _new(store, owner_id=alice)
    legacy = _new(store)
    rec = store.get(owned)
    assert rec is not None and (rec.owner_id, rec.owner_username) == (alice, "alice")
    rec = store.get(legacy)
    assert rec is not None and (rec.owner_id, rec.owner_username) == (None, None)
    # list_by_status carries the owner too (runner recovery reads records through it)
    by_status = {r.id: r.owner_username for r in store.list_by_status(JobState.QUEUED)}
    assert by_status == {owned: "alice", legacy: None}


def test_created_at_has_fixed_precision(store: SqliteJobStore) -> None:
    job_id = _new(store)
    with db.connection(store.db_path) as conn:
        text = conn.execute("SELECT created_at FROM jobs WHERE id = ?", (job_id,)).fetchone()[0]
    assert len(text) == len("2026-10-03T12:00:00.000000+00:00"), text


def test_list_jobs_newest_first_with_keyset_pages(store: SqliteJobStore) -> None:
    ids = [_new(store) for _ in range(5)]
    for i, job_id in enumerate(ids):
        _set_created(store, job_id, f"2026-10-03T12:00:0{i}.000000+00:00")
    expected = list(reversed(ids))
    assert _page_ids(store) == expected

    seen: list[str] = []
    cursor: str | None = None
    pages = 0
    while True:
        page = store.list_jobs(limit=2, cursor=cursor)
        seen += [r.id for r in page.items]
        pages += 1
        cursor = page.next_cursor
        if cursor is None:
            break
    assert seen == expected and pages == 3
    # exactly `limit` rows left -> no next cursor
    assert store.list_jobs(limit=5).next_cursor is None
    assert store.list_jobs(limit=4).next_cursor is not None


def test_list_jobs_ties_on_created_at_order_by_id(store: SqliteJobStore) -> None:
    ids = [_new(store) for _ in range(4)]
    for job_id in ids:
        _set_created(store, job_id, "2026-10-03T12:00:00.000000+00:00")
    expected = sorted(ids, reverse=True)
    first = store.list_jobs(limit=3)
    second = store.list_jobs(limit=3, cursor=first.next_cursor)
    assert [r.id for r in first.items] + [r.id for r in second.items] == expected
    assert second.next_cursor is None


def test_list_jobs_orders_legacy_timestamps_without_microseconds(store: SqliteJobStore) -> None:
    # isoformat() omits ".000000" when microsecond == 0; such legacy text must still sort right
    a, b, c = _new(store), _new(store), _new(store)
    _set_created(store, a, "2026-10-03T12:00:00+00:00")
    _set_created(store, b, "2026-10-03T12:00:00.000001+00:00")
    _set_created(store, c, "2026-10-03T11:59:59.999999+00:00")
    assert _page_ids(store) == [b, a, c]


def test_list_jobs_filters(store: SqliteJobStore) -> None:
    alice = insert_user(store.db_path, "alice")
    bob = insert_user(store.db_path, "bob")
    a1, a2 = _new(store, owner_id=alice), _new(store, owner_id=alice)
    b1 = _new(store, owner_id=bob)
    legacy = _new(store)
    store.mark_succeeded(a2)

    assert set(_page_ids(store, owner_id=alice)) == {a1, a2}
    assert _page_ids(store, owner_id=bob) == [b1]
    assert set(_page_ids(store)) == {a1, a2, b1, legacy}  # admin view includes legacy jobs
    assert _page_ids(store, owner_id=alice, status=JobState.SUCCEEDED) == [a2]
    assert set(_page_ids(store, status=JobState.QUEUED)) == {a1, b1, legacy}
    assert _page_ids(store, owner_id=uuid.uuid4().hex) == []
    page = store.list_jobs(owner_id=alice)
    assert {r.owner_username for r in page.items} == {"alice"}


def test_list_jobs_rejects_bad_cursor_and_limit(store: SqliteJobStore) -> None:
    _new(store)
    for bad in ("!!!", "bm90LWEtY3Vyc29y", encode_cursor("yesterday", "x"), "w4k=", "é"):
        with pytest.raises(InvalidCursor):
            store.list_jobs(cursor=bad)
    for limit in (0, -1, 201):
        with pytest.raises(ValueError):
            store.list_jobs(limit=limit)
    assert len(store.list_jobs(limit=200).items) == 1


def test_cursor_round_trip() -> None:
    created, job_id = "2026-10-03T12:00:00.123456+00:00", uuid.uuid4().hex
    cursor = encode_cursor(created, job_id)
    assert "|" not in cursor and "/" not in cursor and "+" not in cursor
    assert decode_cursor(cursor) == (created, job_id)


def test_delete(store: SqliteJobStore) -> None:
    keep, gone = _new(store), _new(store)
    assert store.delete(gone) is True
    assert store.get(gone) is None
    assert store.delete(gone) is False
    assert store.get(keep) is not None
    # writes to a deleted job are silent no-ops (runner deleted-job guard relies on this)
    store.update_progress(gone, Stage.MEASURING, 0.5)
    store.mark_failed(gone, ErrorCode.INTERNAL_ERROR, "x")
    store.mark_succeeded(gone)
    assert store.get(gone) is None


def test_delete_by_owner(store: SqliteJobStore) -> None:
    alice = insert_user(store.db_path, "alice")
    bob = insert_user(store.db_path, "bob")
    a1, a2 = _new(store, owner_id=alice), _new(store, owner_id=alice)
    b1, legacy = _new(store, owner_id=bob), _new(store)
    assert sorted(store.delete_by_owner(alice)) == sorted([a1, a2])
    assert store.get(a1) is None and store.get(a2) is None
    assert store.get(b1) is not None and store.get(legacy) is not None
    assert store.delete_by_owner(alice) == []


def test_claim_orphans(store: SqliteJobStore) -> None:
    admin = insert_user(store.db_path, "admin", "admin")
    bob = insert_user(store.db_path, "bob")
    legacy = [_new(store), _new(store)]
    owned = _new(store, owner_id=bob)
    assert store.claim_orphans(admin) == 2
    assert all(store.get(j).owner_id == admin for j in legacy)  # type: ignore[union-attr]
    assert store.get(owned).owner_id == bob  # type: ignore[union-attr]
    assert store.claim_orphans(admin) == 0
