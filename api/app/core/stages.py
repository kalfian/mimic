"""Job lifecycle: status values, pipeline stages, UI labels and progress windows (PLAN §4.4).

Progress is a single float 0..1 for the whole job. Each stage owns a window
``[start, end]``; a stage reports its *local* fraction and :func:`overall_progress`
maps it into the window. Windows are contiguous and ordered as :data:`STAGE_ORDER`.
"""

from __future__ import annotations

from enum import StrEnum


class JobState(StrEnum):
    """Top-level job status: ``queued -> processing -> succeeded | failed``."""

    QUEUED = "queued"
    PROCESSING = "processing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"

    @property
    def is_terminal(self) -> bool:
        return self in (JobState.SUCCEEDED, JobState.FAILED)


class Stage(StrEnum):
    """Pipeline stage. ``done`` is set together with ``succeeded``.

    On failure the stage stays at the stage that failed, so the UI can say where.
    """

    QUEUED = "queued"
    PROBING = "probing"
    PREVIEW = "preview"
    SCANNING = "scanning"
    DECODING = "decoding"
    DETECTING_ELEMENTS = "detecting_elements"
    MEASURING = "measuring"
    FITTING = "fitting"
    INTERPRETING = "interpreting"
    GENERATING = "generating"
    DONE = "done"


#: Execution order. ``queued`` first, ``done`` last.
STAGE_ORDER: tuple[Stage, ...] = tuple(Stage)

#: Human label shown in the UI (``JobStatus.stage_label``).
STAGE_LABELS: dict[Stage, str] = {
    Stage.QUEUED: "Waiting",
    Stage.PROBING: "Reading video",
    Stage.PREVIEW: "Preparing preview",
    Stage.SCANNING: "Finding UI states",
    Stage.DECODING: "Extracting frames",
    Stage.DETECTING_ELEMENTS: "Detecting changed elements",
    Stage.MEASURING: "Measuring motion",
    Stage.FITTING: "Estimating timing & easing",
    Stage.INTERPRETING: "Labeling elements (AI)",
    Stage.GENERATING: "Writing specification",
    Stage.DONE: "Done",
}

#: Overall-progress window per stage (inclusive start, inclusive end).
STAGE_WINDOWS: dict[Stage, tuple[float, float]] = {
    Stage.QUEUED: (0.00, 0.00),
    Stage.PROBING: (0.00, 0.05),
    Stage.PREVIEW: (0.05, 0.12),
    Stage.SCANNING: (0.12, 0.30),
    Stage.DECODING: (0.30, 0.42),
    Stage.DETECTING_ELEMENTS: (0.42, 0.52),
    Stage.MEASURING: (0.52, 0.72),
    Stage.FITTING: (0.72, 0.78),
    Stage.INTERPRETING: (0.78, 0.95),
    Stage.GENERATING: (0.95, 1.00),
    Stage.DONE: (1.00, 1.00),
}


def stage_label(stage: Stage | str) -> str:
    """UI label for a stage value."""
    return STAGE_LABELS[Stage(stage)]


def overall_progress(stage: Stage | str, fraction: float = 0.0) -> float:
    """Map a stage-local fraction (clamped to 0..1) into overall job progress 0..1."""
    start, end = STAGE_WINDOWS[Stage(stage)]
    f = min(max(fraction, 0.0), 1.0)
    return start + (end - start) * f
