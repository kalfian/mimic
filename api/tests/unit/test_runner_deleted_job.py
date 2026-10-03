"""Runner deleted-job guard (PLAN-auth A14, §14).

A job's row and directory can be deleted while it is queued or running (``DELETE
/api/jobs/{id}``, an admin deleting its owner). The runner must then write no terminal status and
no ``result.json``, remove the job directory again if the pipeline re-created it, and never raise.
Covers the upload pipeline (every outcome), the interpretation re-run, and a job deleted while
still queued.
"""

from __future__ import annotations

import logging
import threading
import uuid
from collections.abc import Callable, Iterator
from typing import Any

import pytest

from app.api.schemas import JobOptions, ResultEnvelope
from app.config import Settings
from app.core.errors import ErrorCode, PipelineError
from app.core.jobstore import SqliteJobStore
from app.core.runner import JobContext, JobInterrupted, JobRunner, ProgressReporter
from app.core.stages import JobState, Stage
from app.core.storage import LocalJobStorage
from tests.conftest import load_sample_result

PipelineFn = Callable[[JobContext, ProgressReporter], ResultEnvelope]


class SpyStore:
    """Delegates to a real store and records terminal status writes."""

    def __init__(self, inner: SqliteJobStore) -> None:
        self._inner = inner
        self.terminal_writes: list[tuple[str, str]] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def mark_succeeded(self, job_id: str, **kw: Any) -> None:
        self.terminal_writes.append(("succeeded", job_id))
        self._inner.mark_succeeded(job_id, **kw)

    def mark_failed(self, job_id: str, code: ErrorCode | str, message: str) -> None:
        self.terminal_writes.append(("failed", job_id))
        self._inner.mark_failed(job_id, code, message)


@pytest.fixture
def storage(settings: Settings) -> LocalJobStorage:
    s = LocalJobStorage(settings.jobs_dir)
    s.init()
    return s


@pytest.fixture
def store(settings: Settings) -> SpyStore:
    inner = SqliteJobStore(settings.db_path)
    inner.init()
    return SpyStore(inner)


@pytest.fixture
def make_runner(
    settings: Settings, store: SpyStore, storage: LocalJobStorage
) -> Iterator[Callable[..., JobRunner]]:
    runners: list[JobRunner] = []

    def factory(pipeline: PipelineFn, reinterpret: PipelineFn | None = None) -> JobRunner:
        runner = JobRunner(
            store=store,  # type: ignore[arg-type]
            storage=storage,
            settings=settings,
            pipeline=pipeline,
            reinterpret=reinterpret,
            max_workers=1,
        )
        runners.append(runner)
        return runner

    yield factory
    for r in runners:
        r.shutdown(wait=True)


def seed(store: SpyStore, storage: LocalJobStorage) -> str:
    job_id = uuid.uuid4().hex
    store.create(job_id, original_filename="clip.mp4", ext="mp4", options=JobOptions())
    storage.artifact_path(job_id, "input.mp4", create_parents=True).write_bytes(b"x")
    return job_id


def envelope(job_id: str) -> ResultEnvelope:
    data = load_sample_result()
    data["job_id"] = data["spec"]["job_id"] = job_id
    return ResultEnvelope.model_validate(data)


def delete_job(store: SpyStore, storage: LocalJobStorage, job_id: str) -> None:
    """What ``DELETE /api/jobs/{id}`` does: row first, then files."""
    assert store.delete(job_id)
    storage.delete_job(job_id)
    assert not storage.exists(job_id)


class Gate:
    """Lets a fake pipeline block until the test has deleted its job."""

    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()

    def block(self) -> None:
        self.started.set()
        assert self.release.wait(10), "test never released the pipeline"


def rewrite_artifacts(ctx: JobContext) -> None:
    """A pipeline writing after the delete re-creates the job dir."""
    path = ctx.storage.artifact_path(ctx.job_id, "keyframes/state_a.png", create_parents=True)
    path.write_bytes(b"\x89PNG")


@pytest.mark.parametrize("outcome", ["success", "pipeline_error", "crash", "interrupted"])
def test_job_deleted_mid_run(
    make_runner: Callable[..., JobRunner],
    store: SpyStore,
    storage: LocalJobStorage,
    caplog: pytest.LogCaptureFixture,
    outcome: str,
) -> None:
    gate = Gate()

    def pipeline(ctx: JobContext, reporter: ProgressReporter) -> ResultEnvelope:
        reporter.enter(Stage.MEASURING)
        gate.block()
        rewrite_artifacts(ctx)
        reporter.enter(Stage.GENERATING)  # progress write to a deleted row: a no-op
        if outcome == "pipeline_error":
            raise PipelineError(ErrorCode.NO_MOTION_DETECTED)
        if outcome == "crash":
            raise ZeroDivisionError("boom")
        if outcome == "interrupted":
            raise JobInterrupted(ctx.job_id)
        return envelope(ctx.job_id)

    runner = make_runner(pipeline)
    job_id = seed(store, storage)
    future = runner.submit(job_id)
    assert gate.started.wait(10)
    delete_job(store, storage, job_id)

    with caplog.at_level(logging.INFO, logger="app.core.runner"):
        gate.release.set()
        assert future.result(timeout=10) is None  # never raises

    assert not storage.exists(job_id), "job dir re-created by the pipeline was left behind"
    assert store.get(job_id) is None, "deleted row was resurrected"
    assert store.terminal_writes == []
    assert "deleted during the run" in caplog.text
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


def test_rerun_deleted_mid_run(
    make_runner: Callable[..., JobRunner], store: SpyStore, storage: LocalJobStorage
) -> None:
    gate = Gate()

    def reinterpret(ctx: JobContext, reporter: ProgressReporter) -> ResultEnvelope:
        reporter.enter(Stage.INTERPRETING)
        gate.block()
        rewrite_artifacts(ctx)
        return envelope(ctx.job_id)

    runner = make_runner(lambda ctx, r: envelope(ctx.job_id), reinterpret)
    job_id = seed(store, storage)
    storage.write_json(job_id, "result.json", envelope(job_id).model_dump(mode="json"))
    store.mark_succeeded(job_id)
    store.terminal_writes.clear()

    rec = runner.submit_reinterpret(job_id, JobOptions(use_interpreter=True))
    assert rec.status is JobState.QUEUED
    assert gate.started.wait(10)
    delete_job(store, storage, job_id)
    gate.release.set()
    runner.shutdown(wait=True)  # the re-run makes no progress call after release

    assert not storage.exists(job_id)
    assert store.get(job_id) is None
    assert store.terminal_writes == []  # neither the new result nor the "restore previous"


def test_job_deleted_while_queued(
    make_runner: Callable[..., JobRunner], store: SpyStore, storage: LocalJobStorage
) -> None:
    """The second job is deleted before it starts. Its row is gone but (as if the route's file
    removal had failed) its dir is still there: the runner skips it and removes the dir."""
    gate = Gate()
    ran: list[str] = []

    def pipeline(ctx: JobContext, reporter: ProgressReporter) -> ResultEnvelope:
        ran.append(ctx.job_id)
        if len(ran) == 1:
            gate.block()
        return envelope(ctx.job_id)

    runner = make_runner(pipeline)
    first, second = seed(store, storage), seed(store, storage)
    f1, f2 = runner.submit(first), runner.submit(second)
    assert gate.started.wait(10)
    assert store.delete(second)  # row only; the dir stays
    assert storage.exists(second)
    gate.release.set()
    f1.result(timeout=10)
    f2.result(timeout=10)

    assert ran == [first]
    assert not storage.exists(second)
    assert storage.exists(first, "result.json")
    assert store.terminal_writes == [("succeeded", first)]  # the guard leaves live jobs alone


def test_rerun_of_deleted_job_is_not_found(
    make_runner: Callable[..., JobRunner],
    store: SpyStore,
    storage: LocalJobStorage,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = make_runner(lambda ctx, r: envelope(ctx.job_id), lambda ctx, r: envelope(ctx.job_id))
    job_id = seed(store, storage)
    store.mark_succeeded(job_id)

    # deleted between the route's lookup and the claim: not_found, not already_running
    def claim_after_delete(jid: str, options: JobOptions) -> bool:
        store.delete(jid)
        return False

    monkeypatch.setattr(store, "claim_rerun", claim_after_delete)
    with pytest.raises(PipelineError) as exc:
        runner.submit_reinterpret(job_id, JobOptions(use_interpreter=True))
    assert exc.value.code is ErrorCode.NOT_FOUND

    with pytest.raises(PipelineError) as exc:  # already gone before the call
        runner.submit_reinterpret(job_id, JobOptions(use_interpreter=True))
    assert exc.value.code is ErrorCode.NOT_FOUND
