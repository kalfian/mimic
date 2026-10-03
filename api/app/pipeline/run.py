"""Pipeline orchestrator (PLAN §2): video → measurements → interpretation → IR → outputs.

Stage order (progress windows in ``core/stages.py``)::

    probing    A0 probe (+ probe.json)                 preview   A1 H.264 preview (non-fatal)
    scanning   A2 pass 1: energy, cursor, segments     decoding  A3 pass 2 windows (+ crop regrow)
    detecting  A4 state frames, element decomposition  measuring A5 geometry/photometric/shadow
    fitting    A6 timing/easing + confidence, A8 heuristic interaction, A9 keyframes
    interpreting  Layer B (claude_cli / openai_compat / fallback), progress creeps while waiting
    generating    C assemble IR (``assemble.py``) + D ``render_all``

:func:`run` is the job-runner entry point (``PipelineFn``). :func:`analyze` is the same pipeline
without the job store / preview, used by ``scripts/analyze.py`` and ``scripts/eval_synth.py``.

Measurement policy decided in Phase 4 (see PLAN "Phase 4 notes"):

* Insignificant changes are dropped by the §6.7 gates before fitting.
* Appearing/disappearing elements with an anisotropic scale (``scaleX``/``scaleY`` present) drop
  every scale property: while the element is nearly transparent ECC cannot see its size, so the
  extrapolated start scale is a tracking artefact (S8 produced ±3 % fake scales).
  Regular elements that report ``scaleX``/``scaleY`` drop the redundant uniform ``scale``.
* Siblings pushed by an expand/collapse (``caused_by_resize``) only report ``translateY``: the
  resize tag already requires a pure vertical shift, and their flat card surfaces make ECC's
  scale estimate noisy.
* Strong ease-out fits (the curve reaches 95 % early) get a lower timing confidence: their tail
  lies in the measurement noise, so the duration is uncertain (§6.8 + ``ConfidenceParams.tail_*``).
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import numpy as np

from app import PIPELINE_VERSION
from app.api.schemas import Artifacts, KeyframeArtifact, ResultEnvelope
from app.config import Settings
from app.core.errors import RERUN_UNAVAILABLE_MESSAGE, ErrorCode, PipelineError
from app.core.runner import JobContext, ProgressReporter
from app.core.stages import Stage
from app.generate import render_all
from app.interpret import get_interpreter
from app.interpret.base import InterpretationResult, Interpreter
from app.interpret.fallback import FallbackInterpreter
from app.models.ir import (
    CursorEvent,
    ElementStatic,
    MeasuredNumber,
    MeasuredShadow,
    PxValue,
    RatioValue,
    SegmentId,
    SpecWarning,
)
from app.models.measure import (
    ClassifyFeatures,
    CursorTrack,
    ElementCandidate,
    FittedTransition,
    ProbeInfo,
    PropertySeries,
    Rect,
    Scale,
)
from app.pipeline import assemble as asm
from app.pipeline.assemble import Measurement, SegmentWindow, warning
from app.pipeline.classify import classify, cursor_features
from app.pipeline.confidence import (
    cap_for,
    make_transition_confidence,
    static_confidence,
    transition_confidence,
)
from app.pipeline.debug import DebugDump
from app.pipeline.decode import (
    clipped_sides,
    crop_rect_css,
    decode_windows,
    expand_crop,
    frame_size_css,
    plan_windows,
)
from app.pipeline.easing import eval_curve
from app.pipeline.keyframes import KeyframeFile, write_keyframes
from app.pipeline.params import DEFAULT_PARAMS, MeasureParams
from app.pipeline.persist import (
    MEASUREMENT_FILE,
    MeasurementFormatError,
    dump_measurement,
    load_measurement,
)
from app.pipeline.photometric import measure_photometric, refine_appear_endpoints
from app.pipeline.preview import make_preview
from app.pipeline.probe import probe, probe_to_json, probe_warnings, resolve_scale
from app.pipeline.radius import measure_radii
from app.pipeline.regions import detect_elements, soft_change_mask, state_frames
from app.pipeline.scan import scan_video
from app.pipeline.shadow import measure_shadows
from app.pipeline.timing import (
    TimingFit,
    fit_series,
    is_significant,
    normalize_progress,
    timing_resolution_ms,
)
from app.pipeline.track import (
    derive_height_series,
    measure_geometry,
    retrack_faded,
    segment_spans,
)

log = logging.getLogger(__name__)

SEGMENT_ORDER: tuple[SegmentId, ...] = ("fwd", "rev", "rt_in", "rt_out")
FORWARD_SEGMENTS = frozenset({"fwd", "rt_in"})
SCALE_PROPS = frozenset({"scale", "scaleX", "scaleY"})
PROPERTY_ORDER: tuple[str, ...] = (
    "translateX", "translateY", "scale", "scaleX", "scaleY", "height", "opacity",
    "background-color", "color", "box-shadow", "border-radius", "content",
)  # fmt: skip
#: Cursor "below the trigger" test for appearing menus (rule 2): the appear box top must lie
#: 0..this many CSS px below the cursor hotspot at onset, horizontally within the box ± pad.
APPEAR_BELOW_CURSOR_MAX_CSS = 96.0
APPEAR_BELOW_CURSOR_PAD_CSS = 24.0
#: Max cursor events reported in the IR (UI markers).
MAX_CURSOR_EVENTS = 16


# --------------------------------------------------------------------------------------------
# Progress
# --------------------------------------------------------------------------------------------


class Progress(Protocol):
    """The part of :class:`~app.core.runner.ProgressReporter` the pipeline uses."""

    def enter(self, stage: Stage, fraction: float = 0.0) -> None: ...

    def update(self, fraction: float) -> None: ...

    def creeping(
        self, expected_s: float, *, cap: float = 0.95, tick_s: float = 0.5
    ) -> AbstractContextManager[None]: ...


class NullProgress:
    """No-op progress for CLI / eval runs; records per-stage wall time."""

    def __init__(self) -> None:
        self.timings: dict[str, float] = {}
        self._stage: str | None = None
        self._t = time.perf_counter()

    def enter(self, stage: Stage, fraction: float = 0.0) -> None:
        now = time.perf_counter()
        if self._stage is not None:
            self.timings[self._stage] = round(now - self._t, 3)
        self._stage, self._t = str(stage), now

    def update(self, fraction: float) -> None:
        return None

    @contextmanager
    def creeping(
        self, expected_s: float, *, cap: float = 0.95, tick_s: float = 0.5
    ) -> Iterator[None]:
        yield

    def finish(self) -> dict[str, float]:
        self.enter(Stage.DONE)
        return self.timings


# --------------------------------------------------------------------------------------------
# Measurement (Layer A)
# --------------------------------------------------------------------------------------------


@dataclass(slots=True)
class KeyframePlan:
    state_a_s: float
    state_b_s: float
    mid_s: dict[int, float] = field(default_factory=dict)


def _finite_xy(x: float, y: float) -> bool:
    return math.isfinite(x) and math.isfinite(y)


def _inside(r: Rect, x: float, y: float) -> bool:
    return r.x <= x <= r.x2 and r.y <= y <= r.y2


def _el_rect(e: ElementCandidate) -> Rect:
    return e.bbox_a.union(e.bbox_b) if e.bbox_a.area > 0 else e.bbox_b


def hotspot_at(cursor: CursorTrack, t_s: float) -> tuple[float, float] | None:
    """Last known cursor hotspot at or before ``t_s`` (+ half a pass-1 frame)."""
    best = None
    for s in cursor.samples:
        if s.t_s > t_s + 0.017:
            break
        if s.state != "unknown" and _finite_xy(s.x_css, s.y_css):
            best = (s.x_css, s.y_css)
    return best


def _cursor_over(cursor: CursorTrack, r: Rect, t0: float, t1: float) -> bool:
    if not cursor.visible:
        return False
    return any(
        t0 <= s.t_s <= t1 and _finite_xy(s.x_css, s.y_css) and _inside(r, s.x_css, s.y_css)
        for s in cursor.samples
    )


def filter_series(
    series: list[PropertySeries], elements: list[ElementCandidate]
) -> list[PropertySeries]:
    """Drop unreliable / redundant scale series (module docstring)."""
    kind = {e.id: e.kind for e in elements}
    pushed = {e.id for e in elements if "caused_by_resize" in e.notes}
    split = {(s.element_id, s.segment) for s in series if s.property in ("scaleX", "scaleY")}
    out: list[PropertySeries] = []
    for s in series:
        key = (s.element_id, s.segment)
        if s.element_id in pushed and s.property in (*SCALE_PROPS, "translateX"):
            continue  # layout push: a pure vertical shift by construction
        if key in split and s.property in SCALE_PROPS:
            if kind.get(s.element_id) in ("appear", "disappear"):
                continue  # anisotropic scale while invisible: artefact
            if s.property == "scale":
                continue  # scaleX/scaleY carry the change
        out.append(s)
    return out


def tail_x95(bezier: tuple[float, ...]) -> float:
    """Fraction of the duration at which the curve reaches 95 % progress."""
    x = np.linspace(0.0, 1.0, 1001)
    y = eval_curve(bezier, x)
    idx = np.nonzero(y >= 0.95)[0]
    return float(x[idx[0]]) if idx.size else 1.0


def tail_factor(bezier: tuple[float, ...], params: MeasureParams) -> float:
    c = params.confidence
    x95 = tail_x95(bezier)
    f = (x95 - c.tail_x95_zero) / max(c.tail_x95_full - c.tail_x95_zero, 1e-6)
    return float(min(max(f, c.tail_min_factor), 1.0))


EXTRAPOLATED_NOTE = "endpoint extrapolated from opacity (assumes shared timing)"
#: Value-confidence cap for a geometric start/end that was never visible (extrapolated).
EXTRAPOLATED_VALUE_CAP = 0.8


def shared_timing_endpoint(
    s: PropertySeries, op: PropertySeries, op_fit: TimingFit
) -> tuple[float, float] | None:
    """Re-estimate the invisible endpoint of a fading element's geometric series.

    ``refine_appear_endpoints`` regresses the value on the *per-frame* opacity, which carries
    ±0.03 noise (errors-in-variables → biased intercept). Under the same shared-timing
    assumption the fitted opacity curve ``p(t) = E((t − t0) / D)`` is a far less noisy progress
    signal: ``v(t) = v_from + (v_to − v_from)·p(t)`` with the visible end known, one unknown.
    Returns the new ``(v_start, v_end)`` or None when there is too little support.
    """
    x = (np.asarray(s.times, dtype=np.float64) - op_fit.t0_s) / op_fit.duration_s
    p = eval_curve(op_fit.easing.cubic_bezier, np.clip(x, 0.0, 1.0))
    y = np.asarray(s.values, dtype=np.float64)
    start_unknown = op.v_start < 0.1 <= op.v_end  # appears in this segment
    if start_unknown:
        w = 1.0 - p  # y − v_end = (v_start − v_end)·(1 − p)
        if float((w > 0.03).sum()) < 2:
            return None
        d = float(np.sum((y - s.v_end) * w) / max(float(np.sum(w * w)), 1e-12))
        return s.v_end + d, s.v_end
    w = p  # y − v_start = (v_end − v_start)·p
    if float((w > 0.03).sum()) < 2:
        return None
    d = float(np.sum((y - s.v_start) * w) / max(float(np.sum(w * w)), 1e-12))
    return s.v_start, s.v_start + d


def _with_endpoints(s: PropertySeries, v0: float, v1: float) -> PropertySeries:
    cls = PxValue if s.unit == "px" else RatioValue
    return replace(
        s, v_start=v0, v_end=v1, from_value=cls(number=round(v0, 4)),
        to_value=cls(number=round(v1, 4)),
    )  # fmt: skip


def fit_transitions(
    series: list[PropertySeries],
    elements: list[ElementCandidate],
    info: ProbeInfo,
    cursor: CursorTrack,
    windows: dict[str, SegmentWindow],
    params: MeasureParams,
) -> list[FittedTransition]:
    """Gate → joint fit → confidence for every series; ids ``t1..`` in (segment, t0) order.

    Geometric series of fading elements whose invisible endpoint was extrapolated from opacity
    reuse the opacity fit's timing (the same shared-timing assumption that produced the
    endpoint) and get the endpoint re-estimated against the fitted opacity curve.
    """
    kind = {e.id: e.kind for e in elements}
    rect = {e.id: _el_rect(e) for e in elements}
    fitted: list[FittedTransition] = []
    op_fits: dict[tuple[str, str], tuple[PropertySeries, TimingFit]] = {}

    def confidence_for(s: PropertySeries, fit: TimingFit):
        w = windows[s.segment]
        over = s.element_id in rect and _cursor_over(cursor, rect[s.element_id], w.start_s, w.end_s)
        conf = transition_confidence(
            s,
            fit,
            element_kind=kind.get(s.element_id),
            timestamps_estimated=info.timestamps_estimated,
            is_vfr=info.is_vfr,
            cursor_over_element=over,
            params=params.confidence,
        )
        tf = tail_factor(fit.easing.cubic_bezier, params)
        value = conf.value
        if EXTRAPOLATED_NOTE in s.notes:
            value = min(value, EXTRAPOLATED_VALUE_CAP)
        if tf < 1.0 or value != conf.value:
            conf = make_transition_confidence(
                value,
                conf.timing * tf,
                conf.easing,
                cap=cap_for(s.property, kind.get(s.element_id), params.confidence),
                params=params.confidence,
            )
        return conf

    def add(s: PropertySeries, fit: TimingFit) -> None:
        fitted.append(
            FittedTransition(
                id="",
                series=s,
                t0_s=max(fit.t0_s, 0.0),
                duration_s=fit.duration_s,
                easing=fit.easing,
                confidence=confidence_for(s, fit),
                progress_rmse=fit.rmse,
            )
        )

    shared = [s for s in series if EXTRAPOLATED_NOTE in s.notes]
    for s in [s for s in series if EXTRAPOLATED_NOTE not in s.notes]:
        if s.segment not in windows or not is_significant(s, params.fit):
            continue
        fit = fit_series(s, params.fit)
        if fit is None or not math.isfinite(fit.t0_s) or fit.duration_s <= 0:
            log.info("no fit for %s %s %s", s.element_id, s.segment, s.property)
            continue
        if s.property == "opacity":
            op_fits[(s.element_id, s.segment)] = (s, fit)
        add(s, fit)

    for s in shared:
        if s.segment not in windows:
            continue
        op = op_fits.get((s.element_id, s.segment))
        if op is not None:
            ends = shared_timing_endpoint(s, op[0], op[1])
            if ends is not None:
                s = _with_endpoints(s, *ends)
        if not is_significant(s, params.fit):
            continue
        if op is None:
            fit = fit_series(s, params.fit)
            if fit is None or fit.duration_s <= 0:
                continue
            add(s, fit)
            continue
        op_fit = op[1]
        x = (np.asarray(s.times) - op_fit.t0_s) / op_fit.duration_s
        pred = eval_curve(op_fit.easing.cubic_bezier, np.clip(x, 0.0, 1.0))
        prog = normalize_progress(s.values, s.v_start, s.v_end)
        rmse = float(np.sqrt(np.mean((prog - pred) ** 2))) if len(prog) else op_fit.rmse
        fit = replace(op_fit, rmse=max(rmse, op_fit.rmse),
                      easing=op_fit.easing.model_copy(update={"rmse": round(rmse, 4)}))  # fmt: skip
        s.notes.append("timing shared with opacity")
        add(s, fit)

    def order(ft: FittedTransition) -> tuple:
        s = ft.series
        num = int(s.element_id[1:]) if s.element_id[1:].isdigit() else 0
        prop = PROPERTY_ORDER.index(s.property) if s.property in PROPERTY_ORDER else 99
        return (SEGMENT_ORDER.index(s.segment), round(ft.t0_s, 4), num, prop)

    fitted.sort(key=order)
    for i, ft in enumerate(fitted, start=1):
        ft.id = f"t{i}"
    return fitted


def pick_target(
    elements: list[ElementCandidate],
    transitions: list[FittedTransition],
    cursor: CursorTrack,
    onset_s: float,
) -> str | None:
    """Heuristic target: root element under the cursor at onset, else by kind priority."""
    active = {ft.series.element_id for ft in transitions}
    roots = [e for e in elements if e.parent_id is None and e.kind != "backdrop" and e.id in active]
    if not roots:
        roots = [e for e in elements if e.id in active and e.kind != "backdrop"]
    if not roots:
        roots = [e for e in elements if e.id in active] or list(elements)
    if not roots:
        return None
    pos = hotspot_at(cursor, onset_s) if cursor.visible else None
    if pos is not None:
        under = [e for e in roots if _inside(_el_rect(e), *pos)]
        if under:
            return max(under, key=lambda e: _el_rect(e).area).id
    first_t0 = {}
    for ft in transitions:
        eid = ft.series.element_id
        first_t0[eid] = min(first_t0.get(eid, math.inf), ft.t0_s)
    priority = {"resize": 0, "appear": 1, "transform": 2, "photometric": 3}
    best = min(
        roots,
        key=lambda e: (priority.get(e.kind, 9), round(first_t0.get(e.id, math.inf), 3),
                       -_el_rect(e).area),
    )  # fmt: skip
    return best.id


def _appear_below_trigger(
    elements: list[ElementCandidate], cursor_pos: tuple[float, float] | None, gap_css: float
) -> bool:
    appears = [e for e in elements if e.kind == "appear"]
    others = [e for e in elements if e.kind not in ("appear", "backdrop")]
    for a in appears:
        top = a.bbox_b.y
        for o in others:
            gap = top - _el_rect(o).y2
            ox = min(a.bbox_b.x2, _el_rect(o).x2) > max(a.bbox_b.x, _el_rect(o).x)
            if -2.0 <= gap <= gap_css and ox:
                return True
        if cursor_pos is not None:
            x, y = cursor_pos
            pad = APPEAR_BELOW_CURSOR_PAD_CSS
            if a.bbox_b.x - pad <= x <= a.bbox_b.x2 + pad and 0 <= top - y <= (
                APPEAR_BELOW_CURSOR_MAX_CSS
            ):
                return True
    return False


def cursor_events(
    cursor: CursorTrack, target_id: str | None, target: Rect | None
) -> list[CursorEvent]:
    """Cursor timeline for the IR: move_start / stationary flips + enter/leave of the target."""
    if not cursor.visible:
        return []
    events: list[CursorEvent] = []
    prev_state: str | None = None
    prev_in: bool | None = None
    for s in sorted(cursor.samples, key=lambda s: s.t_s):
        t_ms = max(0, int(round(s.t_s * 1000)))
        if s.state == "moving" and prev_state in ("stationary", "unknown"):
            events.append(CursorEvent(t_ms=t_ms, kind="move_start"))
        elif s.state == "stationary" and prev_state == "moving":
            events.append(CursorEvent(t_ms=t_ms, kind="stationary"))
        prev_state = s.state
        if target is not None and s.state != "unknown" and _finite_xy(s.x_css, s.y_css):
            now_in = _inside(target, s.x_css, s.y_css)
            if prev_in is not None and now_in != prev_in:
                events.append(
                    CursorEvent(
                        t_ms=t_ms, kind="enter" if now_in else "leave", element_id=target_id
                    )
                )
            prev_in = now_in
    if len(events) > MAX_CURSOR_EVENTS:
        keep = [e for e in events if e.kind in ("enter", "leave")]
        rest = [e for e in events if e.kind not in ("enter", "leave")]
        events = sorted(keep + rest[: max(0, MAX_CURSOR_EVENTS - len(keep))], key=lambda e: e.t_ms)
    return events[:MAX_CURSOR_EVENTS]


def _event_near(events: list[CursorEvent], kind: str, t_s: float, window_s: float) -> float | None:
    best = None
    for ev in events:
        if ev.kind != kind:
            continue
        d = ev.t_ms / 1000.0 - t_s
        if -window_s <= d <= 0.05 and (best is None or abs(d) < abs(best - t_s)):
            best = ev.t_ms / 1000.0
    return best


def cursor_summary(
    cursor: CursorTrack,
    target_id: str | None,
    events: list[CursorEvent],
    stationary: bool,
    onset_s: float,
) -> str:
    """Plain-language cursor summary; the first element id named is the heuristic target."""
    if not cursor.visible:
        tail = f"; largest changed element {target_id}" if target_id else ""
        return "cursor not visible" + tail
    parts: list[str] = []
    for ev in events:
        if ev.kind in ("enter", "leave") and ev.element_id:
            parts.append(f"{'enters' if ev.kind == 'enter' else 'leaves'} {ev.element_id} "
                         f"at {ev.t_ms} ms")  # fmt: skip
    if not parts and target_id:
        parts.append(f"not over a changed element; motion starts at {target_id}")
    parts.append(f"{'stationary' if stationary else 'moving'} at onset ({int(onset_s * 1000)} ms)")
    return "; ".join(parts)


@dataclass(slots=True)
class MeasureOutput:
    measurement: Measurement
    keyframe_plan: KeyframePlan


def measure(
    video: Path,
    info: ProbeInfo,
    scale: Scale,
    warnings: list[SpecWarning],
    *,
    params: MeasureParams,
    progress: Progress,
    ffmpeg_bin: str = "ffmpeg",
    debug_dir: Path | None = None,
) -> MeasureOutput:
    """Layer A (A2–A8) on one video. Raises ``PipelineError`` for unanalysable recordings."""
    dump = DebugDump(debug_dir)
    warnings = list(warnings)

    # A2 — pass 1
    progress.enter(Stage.SCANNING)
    scan, stats = scan_video(video, info, scale, params, ffmpeg_bin)
    warnings += scan.warnings
    prim = scan.primary
    cursor = scan.cursor

    # A3 — pass 2 (+ crop regrow when a faint element touches the crop edge)
    progress.enter(Stage.DECODING)
    specs, w = plan_windows(scan, info, params)
    warnings += w
    rois = [prim.forward.roi_css] + ([prim.reverse.roi_css] if prim.reverse else [])
    crop_css, crop_px = crop_rect_css(rois, info, scale, params)

    def decode() -> list:
        wfs_, w_ = decode_windows(video, info, scale, specs, crop_css, crop_px, params,
                                  cursor=cursor, pass1_times=scan.frame_times,
                                  ffmpeg_bin=ffmpeg_bin)  # fmt: skip
        return wfs_, w_

    wfs, w = decode()
    warnings += w
    fwd_wf = wfs[0]
    st = state_frames(fwd_wf, prim.forward.start_s, prim.forward.end_s, params,
                      round_trip=prim.round_trip)  # fmt: skip
    progress.update(0.6)
    sides = clipped_sides(soft_change_mask(st.a, st.b, st.valid, params), crop_css, info, scale)
    if sides:
        crop_css, crop_px = expand_crop(crop_css, sides, info, scale, params)
        wfs, _ = decode()
        fwd_wf = wfs[0]
        st = state_frames(fwd_wf, prim.forward.start_s, prim.forward.end_s, params,
                          round_trip=prim.round_trip)  # fmt: skip

    # A4 — elements
    progress.enter(Stage.DETECTING_ELEMENTS)
    rr = detect_elements(fwd_wf, st, frame_size_css(info, scale), params)
    warnings += rr.warnings
    if not rr.elements:
        raise PipelineError(
            ErrorCode.NO_MOTION_DETECTED,
            "Motion was found, but no changed UI element could be isolated. Make sure the "
            "animated element is fully on screen and stands out from its background.",
        )

    # A5 — per-frame measurement
    progress.enter(Stage.MEASURING)
    peak_s: float | None = None
    windows_spans = []
    seg_windows: dict[str, SegmentWindow] = {}
    for wf in wfs:
        if prim.round_trip:
            peak_s = float(fwd_wf.times[st.peak_idx]) if st.peak_idx is not None else None
            if peak_s is None:
                peak_s = (prim.forward.start_s + prim.forward.end_s) / 2
            windows_spans.append(
                (wf, segment_spans(wf, prim.forward.start_s, prim.forward.end_s, peak_s=peak_s))
            )
        else:
            seg = prim.forward if wf.segment_id == "fwd" else prim.reverse
            assert seg is not None
            windows_spans.append((wf, segment_spans(wf, seg.start_s, seg.end_s)))
    if prim.round_trip:
        assert peak_s is not None
        seg_windows["rt_in"] = SegmentWindow("rt_in", prim.forward.start_s, peak_s)
        seg_windows["rt_out"] = SegmentWindow("rt_out", peak_s, prim.forward.end_s)
    else:
        seg_windows["fwd"] = SegmentWindow("fwd", prim.forward.start_s, prim.forward.end_s)
        if prim.reverse is not None:
            seg_windows["rev"] = SegmentWindow("rev", prim.reverse.start_s, prim.reverse.end_s)

    geo = measure_geometry(rr.elements, rr.boxes_px, rr.warps_px, st.a, st.b, st.valid,
                           windows_spans, params)  # fmt: skip
    progress.update(0.4)
    pho = measure_photometric(rr.elements, rr.element_masks, rr.boxes_px, rr.warps_px,
                              rr.overlay, st.a, st.b, st.valid, windows_spans, params,
                              tracks=geo.tracks)  # fmt: skip
    progress.update(0.75)
    sh = measure_shadows(rr.elements, rr.boxes_px, rr.warps_px, rr.element_masks, st.a, st.b,
                         windows_spans, params, tracks=geo.tracks)  # fmt: skip
    rad = measure_radii(rr.elements, rr.boxes_px, rr.warps_px, st.a, st.b, windows_spans,
                        params, tracks=geo.tracks)  # fmt: skip
    retrack_faded(geo, pho.series, rr.elements, rr.boxes_px, st.a, st.b, st.valid,
                  windows_spans, params)  # fmt: skip
    refine_appear_endpoints(geo.series + pho.series)
    heights = derive_height_series(rr.resize_links, geo.series)
    series = geo.series + heights + pho.series + sh.series + rad.series
    if "rotation_detected" in geo.warnings:
        warnings.append(warning("rotation_detected"))

    # A6 — fits + confidence
    progress.enter(Stage.FITTING)
    series = filter_series(series, rr.elements)
    transitions = fit_transitions(series, rr.elements, info, cursor, seg_windows, params)
    fwd_ids = [t for t in transitions if t.series.segment in FORWARD_SEGMENTS]
    if not fwd_ids:
        raise PipelineError(
            ErrorCode.NO_MOTION_DETECTED,
            "Motion was found, but every change was too small to measure reliably. Record at "
            "100 % browser zoom (or zoom in) so the animation covers more pixels.",
        )

    # A8 — heuristic interaction
    fwd_seg = "rt_in" if prim.round_trip else "fwd"
    rev_seg = "rt_out" if prim.round_trip else ("rev" if prim.reverse is not None else None)
    onset = min(t.t0_s for t in transitions if t.series.segment == fwd_seg)
    rev_t = [t for t in transitions if t.series.segment == rev_seg]
    rev_onset = min(t.t0_s for t in rev_t) if rev_t else None
    if prim.round_trip:
        end = max(t.t0_s + t.duration_s for t in transitions)
    else:
        end = max(t.t0_s + t.duration_s for t in fwd_ids)
    target_id = pick_target(rr.elements, transitions, cursor, onset)
    by_id = {e.id: e for e in rr.elements}
    target_rect = _el_rect(by_id[target_id]) if target_id else None
    enter = stationary = leave = False
    if target_rect is not None:
        enter, stationary, leave = cursor_features(
            cursor, target_rect, onset, rev_onset if not prim.round_trip else None, params.classify
        )
    elif cursor.visible:
        _, stationary, _ = cursor_features(cursor, Rect(0, 0, 0, 0), onset, None, params.classify)
    fw, fh = frame_size_css(info, scale)
    active = {t.series.element_id for t in transitions}
    appears = [e for e in rr.elements if e.kind == "appear" and e.id in active]
    target_scale = [t for t in fwd_ids if t.series.element_id == target_id
                    and t.series.property == "scale"]  # fmt: skip
    scale_dir = 0
    if target_scale:
        f, t = target_scale[0].series.from_value, target_scale[0].series.to_value
        scale_dir = int(np.sign(t.number - f.number))  # type: ignore[union-attr]
    features = ClassifyFeatures(
        cursor_visible=cursor.visible,
        enter_before_onset=enter,
        stationary_at_onset=stationary,
        leave_before_reverse=leave,
        has_appear=bool(appears),
        appear_area_frac=max((e.bbox_b.area for e in appears), default=0.0) / max(fw * fh, 1.0),
        has_backdrop=any(e.kind == "backdrop" and e.id in active for e in rr.elements),
        appear_below_trigger=_appear_below_trigger(
            [e for e in rr.elements if e.id in active],
            hotspot_at(cursor, onset) if cursor.visible else None,
            params.classify.appear_below_gap_css,
        ),
        resize_pattern=any(e.kind == "resize" and e.id in active for e in rr.elements),
        round_trip=prim.round_trip,
        forward_reverse=prim.reverse is not None,
        forward_duration_s=max(end - onset, 0.0),
        properties=frozenset(t.series.property for t in fwd_ids),
        scale_direction=scale_dir,
    )
    heuristic = classify(features, params.classify, params.confidence)
    events = cursor_events(cursor, target_id, target_rect)
    # trigger event times (cursor enter / leave), IR segment.trigger_event_ms
    if heuristic.trigger == "pointer_enter":
        seg_windows[fwd_seg].trigger_event_s = _event_near(events, "enter", onset, 0.4)
    if rev_seg and rev_onset is not None and heuristic.reverse_trigger == "pointer_leave":
        seg_windows[rev_seg].trigger_event_s = _event_near(events, "leave", rev_onset, 0.4)

    # statics (low-confidence heuristics)
    statics: dict[str, ElementStatic] = {}
    for eid in {e.id for e in rr.elements}:
        r = rad.static.get(eid)
        shadow_known = eid in sh.static
        if r is None and not shadow_known:
            continue
        statics[eid] = ElementStatic(
            border_radius_px=None
            if r is None
            else MeasuredNumber(
                value=round(r.radius_css, 1),
                confidence=static_confidence(
                    0.5 * r.corners / 4, "border-radius", params.confidence
                ),
            ),  # fmt: skip
            shadow=None
            if not shadow_known
            else MeasuredShadow(
                value=sh.static[eid],
                confidence=static_confidence(
                    0.45 if sh.static[eid] is not None else 0.3, "box-shadow", params.confidence
                ),
            ),
        )

    # A9 plan: stable midpoints + primary transition progress times
    stable_mid = [(s.start_s + s.end_s) / 2 for s in scan.stable]
    a_s = max([m for m in stable_mid if m < prim.forward.start_s], default=0.0)
    if prim.round_trip and peak_s is not None:
        b_s = peak_s
    else:
        limit = prim.reverse.start_s if prim.reverse is not None else math.inf
        b_cands = [m for m in stable_mid if prim.forward.end_s < m < limit]
        b_s = min(b_cands, default=min(prim.forward.end_s + 0.2, info.duration_s - 0.05))
    prim_t = [t for t in fwd_ids if t.series.element_id == target_id] or fwd_ids
    pt = min(prim_t, key=lambda t: t.t0_s)
    mid: dict[int, float] = {}
    xs = np.linspace(0.0, 1.0, 1001)
    ys = eval_curve(pt.easing.cubic_bezier, xs)
    for pct in (25, 50, 75):
        idx = np.nonzero(ys >= pct / 100)[0]
        x = float(xs[idx[0]]) if idx.size else pct / 100
        mid[pct] = pt.t0_s + x * pt.duration_s

    dt_frames = fwd_wf.times[~fwd_wf.is_dup] if fwd_wf.is_dup.any() else fwd_wf.times
    m = Measurement(
        probe=info,
        scale=scale,
        frame_css=(fw, fh),
        direction=heuristic.direction,
        segments=[seg_windows[k] for k in SEGMENT_ORDER if k in seg_windows],
        elements=rr.elements,
        transitions=transitions,
        heuristic=heuristic,
        target_id=target_id,
        cursor_visible=cursor.visible,
        cursor_confidence=cursor.confidence,
        cursor_events=events,
        cursor_summary=cursor_summary(cursor, target_id, events, stationary, onset),
        statics=statics,
        warnings=warnings,
        timing_resolution_ms=timing_resolution_ms(dt_frames),
    )

    if dump.enabled:
        from app.pipeline.debug import element_json, series_json

        dump.energy(scan, stats)
        dump.segments(scan, {"crop_css": [crop_css.x, crop_css.y, crop_css.w, crop_css.h]})
        dump.elements(rr.elements)
        dump.image("state_a", st.a)
        dump.image("state_b", st.b)
        dump.image("change_mask", rr.change_mask)
        dump.series(series)
        dump.write_json("summary.json", {
            "elements": [element_json(e) for e in rr.elements],
            "series": [series_json(s) for s in series],
            "features": {k: getattr(features, k) for k in features.__slots__
                         if k != "properties"},
            "heuristic": {"type": heuristic.type, "trigger": heuristic.trigger,
                          "rule": heuristic.rule, "confidence": heuristic.confidence},
            "target": target_id,
        })  # fmt: skip
    return MeasureOutput(m, KeyframePlan(a_s, b_s, mid))


# --------------------------------------------------------------------------------------------
# Full analysis (no job store)
# --------------------------------------------------------------------------------------------


@dataclass(slots=True)
class Analysis:
    spec: object  # MotionSpec (kept loose to avoid a forward ref in dataclass slots)
    keyframes: list[KeyframeFile]
    interpretation: InterpretationResult
    measurement: Measurement
    #: Job-level warnings handed to ``assemble`` besides the measurement's own.
    extra_warnings: list[SpecWarning] = field(default_factory=list)


def interpreter_expected_s(settings: Settings) -> float:
    """Time constant of the progress creep while waiting on the interpreter."""
    timeout = (
        settings.llm_timeout_s if settings.interpreter == "openai_compat"
        else settings.claude_timeout_s
    )  # fmt: skip
    return timeout / 3


def run_interpreter(
    m: Measurement,
    job_dir: Path,
    keyframes: dict[str, Path],
    interpreter: Interpreter,
    progress: Progress,
    expected_s: float,
) -> InterpretationResult:
    """Layer B on a finished measurement. ``keyframes`` maps artifact names to PNG paths."""
    inp = asm.interpretation_input(m, job_dir, keyframes)
    if isinstance(interpreter, FallbackInterpreter):
        return interpreter.interpret(inp)
    with progress.creeping(expected_s):
        return interpreter.interpret(inp)


def analyze(
    video: Path,
    *,
    settings: Settings,
    job_id: str,
    filename: str,
    job_dir: Path,
    pixel_ratio: str = "auto",
    use_interpreter: bool = False,
    info: ProbeInfo | None = None,
    progress: Progress | None = None,
    params: MeasureParams = DEFAULT_PARAMS,
    keyframes_dir: Path | None = None,
    interpreter: Interpreter | None = None,
    extra_warnings: list[SpecWarning] | None = None,
    pending_warnings: Callable[[], list[SpecWarning]] | None = None,
    debug_dir: Path | None = None,
) -> Analysis:
    """Probe (unless ``info`` is given) → measure → keyframes → interpret → assemble.

    ``pending_warnings`` is called once right before assembling (after Layer B); its warnings
    are added to ``extra_warnings``. The job pipeline uses it to wait for the background preview.
    """
    progress = progress or NullProgress()
    if info is None:
        progress.enter(Stage.PROBING)
        info = probe(video, settings, params)
    scale, scale_warn = resolve_scale(info, pixel_ratio, params)
    base_warnings = scale_warn + probe_warnings(info, params)

    out = measure(video, info, scale, base_warnings, params=params, progress=progress,
                  ffmpeg_bin=settings.ffmpeg_bin, debug_dir=debug_dir)  # fmt: skip
    m = out.measurement

    kfs: list[KeyframeFile] = []
    if keyframes_dir is not None:
        plan = out.keyframe_plan
        kfs = write_keyframes(video, keyframes_dir, info, scale, state_a_s=plan.state_a_s,
                              state_b_s=plan.state_b_s, mid_s=plan.mid_s, elements=m.elements,
                              params=params, ffmpeg_bin=settings.ffmpeg_bin)  # fmt: skip

    progress.enter(Stage.INTERPRETING)
    if interpreter is None:
        interpreter = get_interpreter(settings, use_interpreter)
    interp = run_interpreter(m, job_dir, {k.name: k.path for k in kfs}, interpreter, progress,
                             expected_s=interpreter_expected_s(settings))  # fmt: skip

    extra = list(extra_warnings or [])
    if pending_warnings is not None:
        extra += pending_warnings()
    progress.enter(Stage.GENERATING)
    spec = asm.assemble(
        m,
        interp,
        job_id=job_id,
        filename=filename,
        generated_at=datetime.now(UTC),
        pipeline_version=PIPELINE_VERSION,
        use_interpreter=use_interpreter,
        extra_warnings=extra,
        params=params,
    )
    return Analysis(spec=spec, keyframes=kfs, interpretation=interp, measurement=m,
                    extra_warnings=extra)  # fmt: skip


# --------------------------------------------------------------------------------------------
# Job entry points
# --------------------------------------------------------------------------------------------


class _PreviewCancelled(Exception):  # noqa: N818 - control-flow signal
    pass


class BackgroundPreview:
    """A1 preview transcode on its own thread, overlapping probing and measurement.

    The transcode is a separate (multi-threaded) ffmpeg process, so running it next to the
    mostly single-threaded measurement cut a 15 s 1080p60 job by ~1.2 s (Phase 5 perf pass).
    It reports no progress of its own; :meth:`cancel` (job failed) or ``stopping()`` (runner
    shutdown) kills ffmpeg at its next progress line.
    """

    def __init__(
        self,
        src: Path,
        dst: Path,
        settings: Settings,
        *,
        duration_s: float | None,
        stopping: Callable[[], bool],
    ) -> None:
        self._cancel = threading.Event()
        self._ok = False

        def on_progress(_fraction: float) -> None:
            if self._cancel.is_set() or stopping():
                raise _PreviewCancelled

        def work() -> None:
            try:
                # duration only drives the progress callback (our cancellation hook)
                self._ok = make_preview(src, dst, settings, duration_s=duration_s or 1.0,
                                        on_progress=on_progress)  # fmt: skip
            except _PreviewCancelled:
                self._ok = False
            except Exception:  # never fail the job because of the preview
                log.exception("preview thread failed")
                self._ok = False

        self._thread = threading.Thread(target=work, name="mimic-preview", daemon=True)
        self._thread.start()

    def result(self) -> bool:
        """Wait for the transcode; True if ``preview.mp4`` was written."""
        self._thread.join()
        return self._ok

    def warnings(self) -> list[SpecWarning]:
        return [] if self.result() else [warning("preview_unavailable")]

    def cancel(self) -> None:
        self._cancel.set()
        self._thread.join(timeout=10)


def _envelope(job_id: str, spec: object, artifacts: Artifacts) -> ResultEnvelope:
    outputs = render_all(spec)  # type: ignore[arg-type]
    return ResultEnvelope.model_validate(
        {"job_id": job_id, "spec": spec, "outputs": outputs, "artifacts": artifacts}
    )


def run(ctx: JobContext, reporter: ProgressReporter) -> ResultEnvelope:
    """Job-runner pipeline (``PipelineFn``).

    The preview transcode (A1) runs in the background from the start; the ``preview`` stage is
    therefore not reported on its own (progress goes probing → scanning). Layer A's output is
    stored as ``measurement.json`` for :func:`reinterpret`.
    """
    settings, storage, job_id = ctx.settings, ctx.storage, ctx.job_id
    base_url = f"/api/jobs/{job_id}"

    reporter.enter(Stage.PROBING)
    preview = BackgroundPreview(
        ctx.input_path,
        storage.artifact_path(job_id, "preview.mp4", create_parents=True),
        settings,
        duration_s=ctx.duration_s,
        stopping=lambda: reporter.stopping,
    )
    try:
        # A0 — probe
        info = probe(ctx.input_path, settings)
        storage.write_json(job_id, "probe.json", probe_to_json(info))
        reporter.update(1.0)

        keyframes_dir = storage.artifact_path(job_id, "keyframes/state_a.png", create_parents=True)
        debug_dir = ctx.job_dir / "debug" if settings.debug else None
        result = analyze(
            ctx.input_path,
            settings=settings,
            job_id=job_id,
            filename=ctx.original_filename,
            job_dir=ctx.job_dir,
            pixel_ratio=ctx.options.pixel_ratio,
            use_interpreter=ctx.options.use_interpreter,
            info=info,
            progress=reporter,
            keyframes_dir=keyframes_dir.parent,
            pending_warnings=preview.warnings,
            debug_dir=debug_dir,
        )
    except BaseException:
        preview.cancel()
        raise
    preview_ok = preview.result()
    storage.write_json(job_id, MEASUREMENT_FILE,
                       dump_measurement(result.measurement, result.extra_warnings))  # fmt: skip
    keyframes = [
        KeyframeArtifact(
            name=k.name,
            kind=k.kind,  # type: ignore[arg-type]
            t_ms=None if k.t_s is None else max(0, int(round(k.t_s * 1000))),
            element_id=k.element_id,
            url=f"{base_url}/keyframes/{k.name}",
        )
        for k in result.keyframes
    ]
    artifacts = Artifacts(video_url=f"{base_url}/video" if preview_ok else None,
                          keyframes=keyframes)  # fmt: skip
    envelope = _envelope(job_id, result.spec, artifacts)
    reporter.update(0.8)
    return envelope


def reinterpret(ctx: JobContext, reporter: ProgressReporter) -> ResultEnvelope:
    """Re-run Layer B + assemble + render on a succeeded job's stored measurement.

    ``ctx.options.use_interpreter`` is the new choice (the explicit user action that sends the
    keyframes). Probe, preview, measurement and keyframes are reused from the first run; only
    ``result.json`` (and ``interpretation_raw.json``) change.
    """
    settings, storage, job_id = ctx.settings, ctx.storage, ctx.job_id
    reporter.enter(Stage.INTERPRETING)
    try:
        m, extra = load_measurement(storage.read_json(job_id, MEASUREMENT_FILE))
        previous = ResultEnvelope.model_validate(storage.read_json(job_id, "result.json"))
    except (FileNotFoundError, MeasurementFormatError, ValueError) as exc:
        log.warning("job %s: cannot re-run interpretation: %s", job_id, exc)
        raise PipelineError(ErrorCode.NOT_READY, RERUN_UNAVAILABLE_MESSAGE) from exc
    keyframes = {
        k.name: path
        for k in previous.artifacts.keyframes
        if (path := storage.artifact_path(job_id, f"keyframes/{k.name}")).is_file()
    }
    interpreter = get_interpreter(settings, ctx.options.use_interpreter)
    interp = run_interpreter(m, ctx.job_dir, keyframes, interpreter, reporter,
                             expected_s=interpreter_expected_s(settings))  # fmt: skip

    reporter.enter(Stage.GENERATING)
    spec = asm.assemble(
        m,
        interp,
        job_id=job_id,
        filename=ctx.original_filename,
        generated_at=datetime.now(UTC),
        pipeline_version=PIPELINE_VERSION,
        use_interpreter=ctx.options.use_interpreter,
        extra_warnings=extra,
    )
    envelope = _envelope(job_id, spec, previous.artifacts)
    reporter.update(0.8)
    return envelope
