"""In-process background job runner (PLAN D1, §2, §4.4).

``JobRunner`` executes the pipeline on a ``ThreadPoolExecutor`` (``MIMIC_WORKERS``, default 1),
persists ``result.json`` and the terminal job state. The pipeline callable is injected, so the
runner never imports ``app.pipeline`` (``app.pipeline.run.run`` in production, fakes in tests):

    PipelineFn = Callable[[JobContext, ProgressReporter], ResultEnvelope]

Pipeline code reports progress through :class:`ProgressReporter` and raises
:class:`~app.core.errors.PipelineError` for user-explainable failures. Any other exception
becomes ``internal_error``. On shutdown the reporter raises :class:`JobInterrupted` at the next
progress call so a running job ends as ``failed/interrupted`` instead of blocking the exit.

Re-running interpretation (``POST /api/jobs/{id}/interpret``) uses a second injected callable
with the same signature (``app.pipeline.run.reinterpret``). :meth:`JobRunner.submit_reinterpret`
atomically moves a ``succeeded`` job back to ``queued``; the previous ``result.json`` stays on
disk until the new one replaces it. If the re-run fails or is interrupted the job returns to
``succeeded`` with the previous result (a failure is reported in ``JobStatus.error``).
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from app.api.schemas import JobOptions, ResultEnvelope
from app.config import Settings
from app.core.errors import DEFAULT_MESSAGES, ErrorCode, PipelineError
from app.core.jobstore import JobRecord, JobStore
from app.core.stages import JobState, Stage, overall_progress
from app.core.storage import JobStorage

log = logging.getLogger(__name__)

#: Minimum interval between progress writes within one stage (PLAN §4.3).
PROGRESS_THROTTLE_S = 0.2
#: ``JobStatus.error.message`` of a succeeded job whose interpretation re-run crashed.
RERUN_FAILED = "Re-running AI labeling failed; the previous result is unchanged. Try again."


class JobInterrupted(Exception):  # noqa: N818 - control-flow signal, not an error type
    """Raised inside a job when the runner shuts down."""


@dataclass(frozen=True, slots=True)
class JobContext:
    """Everything a pipeline run needs about its job."""

    job_id: str
    input_path: Path
    job_dir: Path
    original_filename: str
    options: JobOptions
    settings: Settings
    storage: JobStorage
    #: Upload-time duration (``JobSource.duration_s``), if known.
    duration_s: float | None = None


class ProgressReporter:
    """Stage + progress updates for one job (thread-safe, throttled, monotonic).

    * :meth:`enter` switches stage (always written).
    * :meth:`update` sets the stage-local fraction 0..1 (written at most every 200 ms).
    * :meth:`checkpoint` only checks for shutdown.
    * :meth:`creeping` advances the fraction over time while blocking work runs (interpreter).
    """

    def __init__(
        self,
        store: JobStore,
        job_id: str,
        stop: threading.Event | None = None,
        *,
        throttle_s: float = PROGRESS_THROTTLE_S,
    ) -> None:
        self._store = store
        self._job_id = job_id
        self._stop = stop or threading.Event()
        self._throttle_s = throttle_s
        self._lock = threading.Lock()
        self._stage = Stage.QUEUED
        self._progress = 0.0
        self._last_write = -math.inf

    @property
    def stage(self) -> Stage:
        return self._stage

    @property
    def progress(self) -> float:
        """Last overall progress (0..1) reported."""
        return self._progress

    @property
    def stopping(self) -> bool:
        return self._stop.is_set()

    def checkpoint(self) -> None:
        """Raise :class:`JobInterrupted` if the runner is shutting down."""
        if self._stop.is_set():
            raise JobInterrupted(self._job_id)

    def enter(self, stage: Stage, fraction: float = 0.0) -> None:
        """Start ``stage``; written immediately."""
        self.checkpoint()
        with self._lock:
            self._stage = Stage(stage)
            self._write(overall_progress(self._stage, fraction), force=True)

    def update(self, fraction: float) -> None:
        """Progress within the current stage (clamped to 0..1); throttled."""
        self.checkpoint()
        with self._lock:
            self._write(overall_progress(self._stage, fraction), force=False)

    def sleep(self, seconds: float) -> None:
        """Interruptible sleep (test pipelines / polling loops)."""
        if self._stop.wait(max(seconds, 0.0)):
            raise JobInterrupted(self._job_id)

    @contextmanager
    def creeping(
        self, expected_s: float, *, cap: float = 0.95, tick_s: float = 0.5
    ) -> Iterator[None]:
        """Advance the current stage's fraction as ``cap * (1 - exp(-t / expected_s))`` while the
        body runs (e.g. waiting on the ``claude`` subprocess). Never raises from the ticker thread.
        """
        done = threading.Event()
        start = time.monotonic()
        tau = max(expected_s, 1e-3)

        def tick() -> None:
            while not done.wait(tick_s) and not self._stop.is_set():
                frac = cap * (1.0 - math.exp(-(time.monotonic() - start) / tau))
                with self._lock:
                    self._write(overall_progress(self._stage, frac), force=False)

        t = threading.Thread(target=tick, name=f"progress-creep-{self._job_id[:8]}", daemon=True)
        t.start()
        try:
            yield
        finally:
            done.set()
            t.join(timeout=tick_s * 2)

    def _write(self, progress: float, *, force: bool) -> None:
        progress = max(progress, self._progress)  # never move backwards
        now = time.monotonic()
        if not force and now - self._last_write < self._throttle_s:
            self._progress = progress
            return
        try:
            self._store.update_progress(self._job_id, self._stage, progress)
        except Exception:  # progress is best-effort; never fail a job because of it
            log.exception("job %s: progress write failed", self._job_id)
        self._progress = progress
        self._last_write = now


PipelineFn = Callable[[JobContext, ProgressReporter], ResultEnvelope]


class JobRunner:
    """Runs queued jobs on a thread pool and records their terminal state."""

    def __init__(
        self,
        *,
        store: JobStore,
        storage: JobStorage,
        settings: Settings,
        pipeline: PipelineFn,
        reinterpret: PipelineFn | None = None,
        max_workers: int | None = None,
    ) -> None:
        self._store = store
        self._storage = storage
        self._settings = settings
        self._pipeline = pipeline
        self._reinterpret = reinterpret
        self._stop = threading.Event()
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers or settings.workers, thread_name_prefix="mimic-job"
        )

    def submit(self, job_id: str) -> Future[None]:
        """Schedule a queued job. Jobs run in submission order."""
        return self._executor.submit(self._execute, job_id)

    def submit_reinterpret(self, job_id: str, options: JobOptions) -> JobRecord:
        """Re-run Layer B + C + D for a ``succeeded`` job with ``options.use_interpreter``.

        Raises ``already_running`` (409) unless the job is ``succeeded`` right now (checked and
        flipped to ``queued`` atomically, so two clicks cannot start two re-runs).
        """
        if self._reinterpret is None:
            raise PipelineError(ErrorCode.INTERNAL_ERROR, "Re-running interpretation is disabled.")
        before = self._store.get(job_id)
        if before is None:
            raise PipelineError(ErrorCode.NOT_FOUND)
        if not self._store.claim_rerun(job_id, options):
            raise PipelineError(ErrorCode.ALREADY_RUNNING)
        rec = self._store.get(job_id)
        assert rec is not None
        self._executor.submit(self._execute_rerun, job_id, before.options)
        return rec

    def recover(self) -> tuple[list[str], list[str]]:
        """Startup recovery (PLAN §4.4): ``processing`` -> ``failed/interrupted``, ``queued`` ->
        resubmitted (or failed if its upload is gone). Returns ``(interrupted, resubmitted)`` ids.
        """
        interrupted: list[str] = []
        resubmitted: list[str] = []
        for rec in self._store.list_by_status(JobState.PROCESSING):
            if self._storage.exists(rec.id, "result.json"):
                self._restore_result(rec)  # an interrupted re-run: the previous result is intact
                continue
            self._store.mark_failed(
                rec.id, ErrorCode.INTERRUPTED, DEFAULT_MESSAGES[ErrorCode.INTERRUPTED]
            )
            interrupted.append(rec.id)
        for rec in self._store.list_by_status(JobState.QUEUED):
            if self._storage.exists(rec.id, "result.json"):
                self._restore_result(rec)  # pending re-run dropped; result kept
                continue
            if self._storage.exists(rec.id, f"input.{rec.ext}"):
                self.submit(rec.id)
                resubmitted.append(rec.id)
            else:
                self._store.mark_failed(
                    rec.id, ErrorCode.INTERRUPTED, "The uploaded video is missing. Upload it again."
                )
                interrupted.append(rec.id)
        if interrupted or resubmitted:
            log.info(
                "startup recovery: %d interrupted, %d resubmitted",
                len(interrupted),
                len(resubmitted),
            )
        return interrupted, resubmitted

    def shutdown(self, *, wait: bool = True) -> None:
        """Stop accepting work. Queued-but-not-started jobs stay ``queued`` (resubmitted on next
        start); a running job is interrupted at its next progress call.
        """
        self._stop.set()
        self._executor.shutdown(wait=wait, cancel_futures=True)

    # ---- worker -----------------------------------------------------------------------------

    def _execute(self, job_id: str) -> None:
        rec = self._store.get(job_id)
        if rec is None or rec.status is not JobState.QUEUED:
            log.warning("job %s: skipped (status %s)", job_id, rec.status if rec else "missing")
            return
        if self._stop.is_set():
            return  # stays queued; resubmitted on next start

        reporter = ProgressReporter(self._store, job_id, self._stop)
        started = time.monotonic()
        try:
            ctx = self._context(rec, rec.options)
            reporter.enter(Stage.PROBING)
            result = self._pipeline(ctx, reporter)
            if result.job_id != job_id:
                raise RuntimeError(f"pipeline returned a result for job {result.job_id}")
            self._storage.write_json(job_id, "result.json", result.model_dump(mode="json"))
            self._store.mark_succeeded(job_id)
            log.info("job %s: succeeded in %.1fs", job_id, time.monotonic() - started)
        except JobInterrupted:
            log.warning("job %s: interrupted at stage %s", job_id, reporter.stage)
            self._store.mark_failed(
                job_id, ErrorCode.INTERRUPTED, DEFAULT_MESSAGES[ErrorCode.INTERRUPTED]
            )
        except PipelineError as exc:
            log.info("job %s: failed at %s: %s", job_id, reporter.stage, exc)
            self._store.mark_failed(job_id, exc.code, exc.message)
        except Exception:
            log.exception("job %s: internal error at stage %s", job_id, reporter.stage)
            self._store.mark_failed(
                job_id, ErrorCode.INTERNAL_ERROR, DEFAULT_MESSAGES[ErrorCode.INTERNAL_ERROR]
            )

    def _restore_result(self, rec: JobRecord) -> None:
        """Back to ``succeeded`` with the options the stored result was produced with."""
        options = rec.options
        try:
            status = self._storage.read_json(rec.id, "result.json")["spec"]["interpretation"]
            options = options.model_copy(update={"use_interpreter": status["status"] != "disabled"})
        except (OSError, ValueError, KeyError, TypeError):
            log.warning("job %s: could not read result.json options during recovery", rec.id)
        self._store.mark_succeeded(rec.id, options=options)

    def _context(self, rec: JobRecord, options: JobOptions) -> JobContext:
        return JobContext(
            job_id=rec.id,
            input_path=self._storage.input_path(rec.id, rec.ext),
            job_dir=self._storage.job_dir(rec.id),
            original_filename=rec.original_filename,
            options=options,
            settings=self._settings,
            storage=self._storage,
            duration_s=rec.source.duration_s if rec.source is not None else None,
        )

    def _execute_rerun(self, job_id: str, previous: JobOptions) -> None:
        """Run ``reinterpret`` for a claimed job. ``previous`` = the options of the result on
        disk, restored whenever the re-run does not produce a new result."""
        rec = self._store.get(job_id)
        if rec is None or rec.status is not JobState.QUEUED:
            log.warning("job %s: re-run skipped (status %s)", job_id, rec.status if rec else "-")
            return
        assert self._reinterpret is not None
        if self._stop.is_set():
            self._store.mark_succeeded(job_id, options=previous)  # not started: keep result
            return

        reporter = ProgressReporter(self._store, job_id, self._stop)
        started = time.monotonic()
        try:
            result = self._reinterpret(self._context(rec, rec.options), reporter)
            if result.job_id != job_id:
                raise RuntimeError(f"re-run returned a result for job {result.job_id}")
            self._storage.write_json(job_id, "result.json", result.model_dump(mode="json"))
            self._store.mark_succeeded(job_id)
            log.info("job %s: interpretation re-run in %.1fs", job_id, time.monotonic() - started)
        except JobInterrupted:
            log.warning("job %s: re-run interrupted; previous result kept", job_id)
            self._store.mark_succeeded(job_id, options=previous)
        except PipelineError as exc:
            log.info("job %s: re-run failed: %s", job_id, exc)
            self._store.mark_succeeded(job_id, options=previous, error=(exc.code, exc.message))
        except Exception:
            log.exception("job %s: re-run internal error at stage %s", job_id, reporter.stage)
            self._store.mark_succeeded(
                job_id, options=previous, error=(ErrorCode.INTERNAL_ERROR, RERUN_FAILED)
            )
