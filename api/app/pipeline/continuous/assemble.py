"""Continuous measurement + interpretation → IR ``MotionSpec`` (PLAN-continuous §4.7, §5).

* :func:`interpretation_input_continuous`: what Layer B may see — one ``scroller`` element whose
  summary is words only ("scroller; content moves horizontally at about 40 px/s; 4 drags") plus
  the transition-style video meta. The interpreter labels the scroller and writes ``structure``;
  it never supplies numbers.
* :func:`assemble_continuous`: builds the IR 0.2 continuous spec: one ``MotionElement`` of kind
  ``scroller`` (plus any measured child elements already in ``Measurement.elements``), the
  ``continuous`` section (phases, behaviour, decimated samples ≤ ``CONTINUOUS_SAMPLE_MAX``) and
  the interaction block. **Interaction type in continuous mode is always heuristic**
  (``drag`` when a drag phase exists, else ``continuous``); ``merge_type`` is bypassed.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from datetime import datetime
from pathlib import Path

import numpy as np

from app.interpret.base import (
    ElementSummary,
    InterpretationInput,
    InterpretationResult,
    VideoMeta,
)
from app.models.ir import (
    CONTINUOUS_SAMPLE_MAX,
    Autoplay,
    Box,
    CardScale,
    ContinuousMotion,
    Cursor,
    ElementStatic,
    Interaction,
    Interpretation,
    LoopInfo,
    MeasuredNumber,
    Meta,
    MotionElement,
    MotionSpec,
    Phase,
    Source,
    Span,
    SpecWarning,
    TotalDuration,
    Trigger,
)
from app.models.measure import (
    ContinuousMeasurement,
    HeuristicInteraction,
    PhaseCandidate,
    Rect,
    ScrollerAnalysis,
)
from app.pipeline.assemble import Measurement, dedupe_warnings, warning
from app.pipeline.continuous import confidence as cconf
from app.pipeline.continuous.displacement import smoothed_velocity
from app.pipeline.params import DEFAULT_PARAMS, MeasureParams

#: Heuristic interaction-type confidence: drag phases seen / autoplay only.
TYPE_CONF_DRAG = 0.75
TYPE_CONF_AUTOPLAY = 0.7
SCROLLER_ID = "e1"

_CONT_WARNING_TEXT: dict[str, tuple[str, str]] = {
    "cursor_not_visible": (
        "warn",
        "No cursor was visible; what starts and stops the motion is inferred from the motion only.",
    ),
    "extra_scrollers_ignored": (
        "info",
        "Other continuously moving areas were found; only the largest interacted scroller is "
        "described.",
    ),
}


# --------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------


def _ms(seconds: float) -> int:
    return max(0, int(round(seconds * 1000.0)))


def _box(r: Rect) -> Box:
    return Box(
        x=round(r.x, 1), y=round(r.y, 1), w=round(max(r.w, 0.0), 1), h=round(max(r.h, 0.0), 1)
    )


def _r1(v: float) -> float:
    r = round(float(v), 1)
    return 0.0 if r == 0 else r


def _axis_words(axis: str) -> str:
    return "horizontally" if axis == "x" else "vertically"


def _direction(axis: str, v: float) -> str:
    if axis == "x":
        return "right" if v > 0 else "left"
    return "down" if v > 0 else "up"


def _scroller(m: Measurement) -> ScrollerAnalysis:
    if m.continuous is None:
        raise ValueError("assemble_continuous needs Measurement.continuous")
    sa = m.continuous.scroller
    if not sa.is_scroller or sa.series is None or not sa.phases or sa.behavior is None:
        raise ValueError("the chosen scroller has no measured phases")
    return sa


def _region(sa: ScrollerAnalysis) -> Rect:
    assert sa.series is not None
    return sa.series.region


def continuous_heuristic(sa: ScrollerAnalysis) -> HeuristicInteraction:
    """Heuristic interaction for a scroller (also usable by ``run.py`` to fill
    ``Measurement.heuristic``): ``drag`` with drag phases, else ``continuous`` (marquee)."""
    drags = any(p.kind == "drag" for p in sa.phases)
    return HeuristicInteraction(
        type="drag" if drags else "continuous",
        trigger="drag" if drags else "autoplay",
        reverse_trigger="none",
        direction="continuous",
        confidence=TYPE_CONF_DRAG if drags else TYPE_CONF_AUTOPLAY,
        rule=0,
    )


def scroller_summary(sa: ScrollerAnalysis) -> str:
    """Words-only summary for Layer B (rounded speed, counts; no fitted parameters)."""
    parts = ["scroller"]
    axis = sa.axis or "x"
    if sa.autoplay_velocity is not None:
        speed = abs(sa.autoplay_velocity)
        shown = round(speed, -1) if speed >= 100 else round(speed)
        parts.append(
            f"content moves {_axis_words(axis)} ({_direction(axis, sa.autoplay_velocity)}) "
            f"at about {shown:.0f} px/s"
        )
    else:
        parts.append(f"content moves {_axis_words(axis)}")
    n_drag = sum(1 for p in sa.phases if p.kind == "drag")
    if n_drag:
        parts.append(f"{n_drag} drag{'s' if n_drag != 1 else ''}")
    if any(p.kind == "decelerate" for p in sa.phases):
        parts.append("autoplay slows down at least once")
    return "; ".join(parts)


# --------------------------------------------------------------------------------------------
# Interpreter input
# --------------------------------------------------------------------------------------------


def interpretation_input_continuous(
    m: Measurement, job_dir: Path, keyframe_paths: dict[str, Path]
) -> InterpretationInput:
    """Interpreter input for a continuous job (the scroller first, then measured children)."""
    sa = _scroller(m)
    h = continuous_heuristic(sa)
    elements = [
        ElementSummary(
            id=SCROLLER_ID,
            bbox=_box(_region(sa)),
            kind="scroller",
            parent_id=None,
            text_like=False,
            change_summary=scroller_summary(sa),
        )
    ]
    for e in m.elements:
        if e.id == SCROLLER_ID or e.kind == "scroller":
            continue
        elements.append(
            ElementSummary(
                id=e.id,
                bbox=_box(e.bbox_a if e.bbox_a.area > 0 else e.bbox_b),
                kind=e.kind,
                parent_id=e.parent_id,
                text_like=e.text_like,
                change_summary="moves with the scroller content",
            )
        )
    p = m.probe
    cursor = m.cursor_summary or ("no cursor visible" if not m.cursor_visible else "")
    return InterpretationInput(
        job_dir=job_dir,
        keyframe_paths=keyframe_paths,
        elements=elements,
        heuristic_type=h.type,
        heuristic_trigger=h.trigger,
        cursor_summary=cursor,
        video_meta=VideoMeta(
            width=p.width,
            height=p.height,
            duration_ms=max(1, _ms(p.duration_s)),
            pixel_ratio=m.scale.pixel_ratio,
        ),
    )


# --------------------------------------------------------------------------------------------
# IR pieces
# --------------------------------------------------------------------------------------------


def ir_phases(phases: Sequence[PhaseCandidate], span: tuple[int, int]) -> list[Phase]:
    """``PhaseCandidate`` → IR ``Phase`` (ids p1..pN, ms boundaries kept contiguous and
    strictly increasing inside ``span``; zero-length phases after rounding are dropped)."""
    out: list[Phase] = []
    lo, hi = span
    prev_end = lo
    for i, ph in enumerate(phases):
        start = max(_ms(ph.start_s), prev_end)
        end = min(_ms(ph.end_s), hi)
        if i == len(phases) - 1:
            end = hi
        # a phase shorter than 1 ms after rounding keeps 1 ms (dropping it would desync the
        # behaviour counts the IR validates against the phase list)
        end = max(end, start + 1)
        if end > hi:
            continue
        out.append(
            Phase(
                id=f"p{len(out) + 1}",
                kind=ph.kind,
                start_ms=start,
                end_ms=end,
                v_start_px_s=_r1(ph.v_start),
                v_end_px_s=_r1(ph.v_end),
                v_peak_px_s=_r1(ph.v_peak),
                displacement_px=_r1(ph.displacement),
                fit=ph.fit,
                interrupted=ph.interrupted,
                evidence=ph.evidence,
                confidence=cconf.ir_confidence(ph.confidence),
                notes=list(ph.notes),
            )
        )
        prev_end = end
    return out


def ir_samples(
    sa: ScrollerAnalysis, params: MeasureParams = DEFAULT_PARAMS
) -> list[tuple[int, float, float, float]]:
    """``[t_ms, v_px_s, pos_px, quality]`` decimated to ≤ ``CONTINUOUS_SAMPLE_MAX`` points,
    strictly increasing ``t_ms`` (velocity smoothed like the labeller's; position integrated)."""
    s = sa.series
    assert s is not None
    t, x, q = s.times, s.pos, s.quality
    if len(t) == 0:
        return []
    v = smoothed_velocity(t, x, params.continuous.velocity_half_window)
    cap = min(CONTINUOUS_SAMPLE_MAX, params.continuous.samples_max)
    idx = np.arange(len(t))
    if len(t) > cap:
        idx = np.unique(np.round(np.linspace(0, len(t) - 1, cap)).astype(int))
    out: list[tuple[int, float, float, float]] = []
    last = -1
    for i in idx:
        tm = _ms(float(t[i]))
        if tm <= last:
            continue
        qi = float(np.clip(q[i], 0.0, 1.0)) if math.isfinite(float(q[i])) else 0.0
        out.append((tm, _r1(v[i]), round(float(x[i]), 2), round(qi, 3)))
        last = tm
    return out[:cap]


def _autoplay(sa: ScrollerAnalysis) -> Autoplay | None:
    v = sa.autoplay_velocity
    if v is None or abs(v) < 1e-9:
        return None
    axis = sa.axis or "x"
    v = _r1(v)
    if v == 0:
        return None
    speed = abs(v)
    conf = cconf.ir_confidence(sa.autoplay_confidence, DEFAULT_PARAMS.continuous.cap_speed)
    if sa.loop_period_px is not None:
        period = round(sa.loop_period_px, 1)
        loop = LoopInfo(
            observed=True,
            period_px=MeasuredNumber(
                value=period,
                confidence=cconf.ir_confidence(
                    sa.loop_confidence, DEFAULT_PARAMS.continuous.cap_loop
                ),  # fmt: skip
            ),
            duration_ms=MeasuredNumber(
                value=round(period / speed * 1000.0, 1),
                confidence=cconf.ir_confidence(
                    min(sa.loop_confidence, sa.autoplay_confidence),
                    DEFAULT_PARAMS.continuous.cap_loop,
                ),
            ),
        )
    else:
        loop = LoopInfo(observed=False, period_px=None, duration_ms=None)
    return Autoplay(
        direction=_direction(axis, v),  # type: ignore[arg-type]
        speed_px_s=MeasuredNumber(value=speed, confidence=conf),
        velocity_px_s=v,
        easing="linear",
        loop=loop,
    )


def _measured(value: float | None, conf: float, cap: float) -> MeasuredNumber | None:
    if value is None:
        return None
    return MeasuredNumber(value=round(float(value), 1), confidence=cconf.ir_confidence(conf, cap))


def card_scale(
    sa: ScrollerAnalysis, region: Rect | Box, axis: str, params: MeasureParams = DEFAULT_PARAMS
) -> CardScale | None:
    """The census's position-dependent card scale (P2c) as IR, or None for rigid cards.

    ``scale(d) = 1 + k d²`` (``k = card_zoom_per_px2``, d from the scroller centre) is stored
    by its value at the farthest measured card (``card_zoom_reach_px``), plus the mean
    magnification over the scroller ``1 + k half² / 3`` that converts the measured on-screen
    speeds to the unscaled track. Confidence capped at ``cap_card_scale`` (medium)."""
    k, reach = sa.card_zoom_per_px2, sa.card_zoom_reach_px
    if k is None or reach is None or k <= 0 or reach <= 0:
        return None
    half = (region.w if axis == "x" else region.h) / 2.0
    ref = round(min(reach, half), 1)
    if ref <= 0 or half <= 0:
        return None
    at_ref = round(1.0 + k * ref * ref, 4)
    if at_ref <= 1.0:
        return None
    # from the rounded stored values, so the IR's own consistency check holds exactly
    mean = round(1.0 + (at_ref - 1.0) * (half / ref) ** 2 / 3.0, 4)
    return CardScale(
        reference_distance_px=ref,
        scale_at_reference=at_ref,
        mean_scale=mean,
        confidence=cconf.ir_confidence(sa.card_zoom_confidence, params.continuous.cap_card_scale),
    )


def _trigger_description(sa: ScrollerAnalysis) -> str:
    kinds = {p.kind for p in sa.phases}
    parts: list[str] = []
    if sa.autoplay_velocity is not None:
        parts.append("Autoplaying scroller")
    else:
        parts.append("Scroller")
    if "drag" in kinds:
        parts.append("that is dragged by the pointer")
    tail: list[str] = []
    if "inertia" in kinds:
        tail.append("after release it coasts")
    if "snap" in kinds:
        tail.append("then settles on a card")
    if "resume" in kinds:
        tail.append("then autoplay resumes")
    text = " ".join(parts)
    if tail:
        text += "; " + ", ".join(tail)
    return text + "."


# --------------------------------------------------------------------------------------------
# Assemble
# --------------------------------------------------------------------------------------------


def assemble_continuous(
    m: Measurement,
    interp: InterpretationResult,
    *,
    job_id: str,
    filename: str,
    generated_at: datetime,
    pipeline_version: str,
    use_interpreter: bool,
    extra_warnings: Iterable[SpecWarning] = (),
    params: MeasureParams = DEFAULT_PARAMS,
) -> MotionSpec:
    """Build the validated continuous-mode IR (see module docstring)."""
    sa = _scroller(m)
    cm: ContinuousMeasurement = m.continuous  # type: ignore[assignment]
    p = m.probe
    axis = sa.axis or "x"
    region = _region(sa)
    assert sa.series is not None and sa.behavior is not None

    # ---- span + phases ------------------------------------------------------------------------
    t = sa.series.times
    span_lo, span_hi = _ms(float(t[0])), _ms(float(t[-1]))
    if span_hi <= span_lo:
        span_hi = span_lo + 1
    phases = ir_phases(sa.phases, (span_lo, span_hi))
    if not phases:
        raise ValueError("no phase survived millisecond rounding")

    # ---- elements -----------------------------------------------------------------------------
    ie = interp.elements.get(SCROLLER_ID)
    ok = interp.status == "ok" and ie is not None
    scroller_static = m.statics.get(SCROLLER_ID, ElementStatic())
    elements = [
        MotionElement(
            id=SCROLLER_ID,
            label=ie.label if ie else f"Scroller ({SCROLLER_ID})",
            label_source=ie.label_source if ok else "heuristic",  # type: ignore[union-attr]
            role="scroller",
            parent_id=None,
            kind="scroller",
            bbox_initial=_box(region),
            bbox_active=_box(region),
            text_like=False,
            static=scroller_static,
        )
    ]
    known = {SCROLLER_ID}
    for e in m.elements:
        if e.id in known or e.kind == "scroller":
            continue
        iel = interp.elements.get(e.id)
        parent = e.parent_id if e.parent_id in known else SCROLLER_ID
        elements.append(
            MotionElement(
                id=e.id,
                label=iel.label if iel else f"Element ({e.id})",
                label_source=iel.label_source if (interp.status == "ok" and iel) else "heuristic",  # type: ignore[union-attr]
                role=iel.role if iel and iel.role != "scroller" else "other",
                parent_id=parent,
                kind=e.kind,
                bbox_initial=_box(e.bbox_a),
                bbox_active=_box(e.bbox_b),
                text_like=e.text_like,
                static=m.statics.get(e.id, ElementStatic()),
            )
        )
        known.add(e.id)
    label = elements[0].label

    # ---- continuous section ---------------------------------------------------------------------
    c = params.continuous
    continuous = ContinuousMotion(
        element_id=SCROLLER_ID,
        axis=axis,
        region=_box(region),
        region_confidence=cconf.ir_confidence(sa.series.region_confidence, c.cap_region),
        autoplay=_autoplay(sa),
        pitch_px=_measured(sa.pitch_px, sa.pitch_confidence, c.cap_pitch),
        gap_px=_measured(sa.gap_px, sa.gap_confidence, c.cap_pitch),
        card_scale=card_scale(sa, _box(region), axis, params),
        phases=phases,
        behavior=sa.behavior,
        span_ms=Span(start_ms=span_lo, end_ms=span_hi),
        samples=ir_samples(sa, params),
    )

    # ---- interaction ----------------------------------------------------------------------------
    h = continuous_heuristic(sa)
    trigger_desc = _trigger_description(sa)
    if interp.status == "ok" and interp.interaction.trigger_description:
        trigger_desc = interp.interaction.trigger_description
    trig_conf = h.confidence if m.cursor_visible else min(h.confidence, 0.6)
    interaction = Interaction(
        type=h.type,
        type_source="heuristic",
        type_confidence=cconf.ir_confidence(h.confidence),
        pattern="carousel" if h.type == "drag" else "marquee",
        target_element_id=SCROLLER_ID,
        target_label=label,
        target_description=interp.interaction.target_description if interp.status == "ok" else None,
        trigger=Trigger(
            kind=h.trigger,
            reverse_kind="none",
            description=trigger_desc,
            confidence=cconf.ir_confidence(trig_conf),
        ),
        direction="continuous",
        total_duration_ms=TotalDuration(forward=span_hi - span_lo, reverse=None),
    )

    # ---- warnings -------------------------------------------------------------------------------
    warns: list[SpecWarning] = list(m.warnings) + list(sa.warnings) + list(extra_warnings)
    if not m.cursor_visible:
        sev, text = _CONT_WARNING_TEXT["cursor_not_visible"]
        warns.append(warning("cursor_not_visible", text, sev))
    if cm.extra_scrollers and not any(w.code == "extra_scrollers_ignored" for w in warns):
        sev, text = _CONT_WARNING_TEXT["extra_scrollers_ignored"]
        warns.append(warning("extra_scrollers_ignored", text, sev))
    if use_interpreter and interp.status in ("fallback", "timeout", "error"):
        reason = {"timeout": "timed out", "error": "failed", "fallback": "was unavailable"}
        warns.append(
            warning(
                "interpretation_fallback",
                f"AI labeling {reason[interp.status]}; element names and roles are heuristic.",
            )
        )

    events = [ev for ev in m.cursor_events if ev.element_id is None or ev.element_id in known]
    return MotionSpec(
        job_id=job_id,
        meta=Meta(generated_at=generated_at, pipeline_version=pipeline_version),
        mode="continuous",
        scene=m.scene,
        source=Source(
            filename=filename,
            container=p.container,
            codec=p.codec,
            width=p.width,
            height=p.height,
            duration_ms=max(1, _ms(p.duration_s)),
            fps_nominal=round(p.fps_nominal if p.fps_nominal > 0 else p.fps_effective, 3),
            fps_effective=round(max(p.fps_effective, 1e-3), 3),
            is_vfr=p.is_vfr,
            pixel_ratio=m.scale.pixel_ratio,
            pixel_ratio_source=m.scale.pixel_ratio_source,
            timing_resolution_ms=max(1, m.timing_resolution_ms),
        ),
        interaction=interaction,
        elements=elements,
        segments=[],
        transitions=[],
        relationships=[],
        continuous=continuous,
        cursor=Cursor(
            visible=m.cursor_visible,
            confidence=round(min(max(m.cursor_confidence, 0.0), 1.0), 3),
            events=events,
        ),
        structure=list(interp.structure),
        interpretation=Interpretation(
            provider=interp.provider,
            model=interp.model,
            status=interp.status,
            duration_ms=interp.duration_ms,
        ),
        warnings=dedupe_warnings(warns),
    )


__all__ = [
    "SCROLLER_ID",
    "assemble_continuous",
    "card_scale",
    "continuous_heuristic",
    "interpretation_input_continuous",
    "ir_phases",
    "ir_samples",
    "scroller_summary",
]
