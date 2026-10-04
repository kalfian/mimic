"""Measurements + interpretation → IR ``MotionSpec`` (PLAN §2 step C, §7.5, §8).

``run.py`` produces a :class:`Measurement` (Layer A: every number, the heuristic interaction
and the heuristic target) and an ``InterpretationResult`` (Layer B: labels only). This module

* builds the interpreter input (:func:`interpretation_input`): element change summaries in words,
  cursor summary whose first element id is the heuristic target (the fallback reads it), and
* merges both into a validated :class:`~app.models.ir.MotionSpec` (:func:`assemble`).

Merge policy (§7.5):

* Labels / roles / structure come from the interpretation result as-is: an ``ok`` result already
  carries heuristic labels for elements the model omitted (``label_source`` per element).
* Interaction type: the interpreter overrides the heuristic only when its status is ``ok``, its
  ``type_confidence ≥ 0.7`` and (heuristic confidence < 0.8 or they agree). Both confident and
  disagreeing → keep the heuristic + warning ``interpretation_disagrees``. When the status is not
  ``ok`` the heuristic confidence is used (fallback results report ``type_confidence = 0``).
* Trigger kind is never taken from the interpreter (cursor evidence only); its prose description
  is, when the status is ``ok``.
* Hierarchy: an interpreter parent replaces the CV parent only if the child box lies ≥ 70 %
  inside the new parent **and** neither the old nor the new parent has translate/scale
  transitions. Child transforms are measured relative to the CV parent, so re-parenting under a
  moving element would silently change what the numbers mean.
* Numbers are never read from the interpreter (D4).

Measured appearance (PLAN-continuous §13, Track A): :func:`apply_appearance` maps an
``appearance.AppearanceResult`` into the measurement — per-element ``background_color`` /
``text_color`` / ``font_size_px`` merged into ``Measurement.statics`` and the top-level
``Measurement.scene`` — so it is persisted with the measurement (re-runs keep it). Without that
call nothing changes: both stay empty / ``None`` and the IR omits them.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from app.interpret.base import (
    ElementSummary,
    InterpretationInput,
    InterpretationResult,
    VideoMeta,
)
from app.models.ir import (
    Box,
    ColorValue,
    Cursor,
    CursorEvent,
    ElementStatic,
    Interaction,
    InteractionType,
    Interpretation,
    MeasuredColor,
    MeasuredNumber,
    Meta,
    MotionElement,
    MotionSpec,
    PxValue,
    RatioValue,
    Scene,
    Segment,
    SegmentId,
    Size,
    Source,
    SpecWarning,
    TotalDuration,
    Transition,
    Trigger,
    Value,
)
from app.models.measure import (
    ContinuousMeasurement,
    ElementCandidate,
    FittedTransition,
    HeuristicInteraction,
    ProbeInfo,
    Rect,
    Scale,
)
from app.pipeline.appearance import (
    FONT_SIZE_CONF_CAP,
    AppearanceResult,
    ColorEstimate,
    FontSizeEstimate,
)
from app.pipeline.classify import pattern_for
from app.pipeline.confidence import confidence as make_confidence
from app.pipeline.confidence import static_confidence
from app.pipeline.params import DEFAULT_PARAMS, MeasureParams
from app.pipeline.relationships import TimedTransition, relationships
from app.pipeline.timing import normalize_progress, progress_samples

#: Interpreter type override thresholds (§7.5).
INTERPRETER_TYPE_MIN_CONF = 0.7
HEURISTIC_CONFIDENT = 0.8
#: Interpreter re-parenting needs this much of the child inside the new parent (§7.5).
REPARENT_MIN_CONTAINMENT = 0.7

_SCALE_PROPS = frozenset({"scale", "scaleX", "scaleY"})
GEOMETRIC_PROPS = frozenset({"translateX", "translateY", "scale", "scaleX", "scaleY"})
FORWARD_SEGMENTS: tuple[SegmentId, ...] = ("fwd", "rt_in")
SEGMENT_ORDER: tuple[SegmentId, ...] = ("fwd", "rev", "rt_in", "rt_out")

_WARNING_TEXT: dict[str, tuple[str, str]] = {
    "cursor_not_visible": (
        "info",
        "No cursor was found in the recording; the trigger is guessed from the motion shape.",
    ),
    "reverse_not_recorded": (
        "info",
        "Only the forward motion was recorded; reverse timings are not measured.",
    ),
    "rotation_detected": (
        "warn",
        "An element appears to rotate; rotation is not measured and may distort other values.",
    ),
    "interpretation_fallback": (
        "warn",
        "AI labeling did not complete; element names and roles are heuristic.",
    ),
    "preview_unavailable": (
        "warn",
        "The preview video could not be generated; showing keyframes instead.",
    ),
}

#: Internal series notes → user-facing transition notes (None = drop).
_NOTE_MAP: dict[str, str | None] = {
    "caused_by_resize": "caused_by_resize (pushed by the height change above it)",
    "opacity_from_dimming": "estimated from dimming; opacity vs color is ambiguous here",
    "below_delta_e": None,
    "shadow increases": None,
    "shadow decreases": None,
}


# --------------------------------------------------------------------------------------------
# Data produced by run.py
# --------------------------------------------------------------------------------------------


@dataclass(slots=True)
class SegmentWindow:
    """Detected active run of one IR segment (absolute seconds)."""

    id: SegmentId
    start_s: float
    end_s: float
    trigger_event_s: float | None = None


@dataclass(slots=True)
class Measurement:
    """Everything Layer A measured for one video (input of :func:`assemble`)."""

    probe: ProbeInfo
    scale: Scale
    frame_css: tuple[float, float]
    direction: str  # IR Direction
    segments: list[SegmentWindow]
    elements: list[ElementCandidate]
    transitions: list[FittedTransition]  # ids assigned, sorted
    heuristic: HeuristicInteraction
    target_id: str | None
    cursor_visible: bool
    cursor_confidence: float
    cursor_events: list[CursorEvent]
    cursor_summary: str
    statics: dict[str, ElementStatic] = field(default_factory=dict)
    warnings: list[SpecWarning] = field(default_factory=list)
    timing_resolution_ms: int = 17
    #: Continuous mode only (PLAN-continuous §4.7); None for transition jobs. Persisted files
    #: written before this field existed decode with the default (``persist.FORMAT_VERSION`` 1).
    continuous: ContinuousMeasurement | None = None
    #: Recorded page context (PLAN-continuous §13), set by :func:`apply_appearance`; None = not
    #: measured (IR ``scene`` omitted). An IR model, so ``persist`` stores it without changes;
    #: files written before this field existed decode with the default.
    scene: Scene | None = None


# --------------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------------


def _ms(seconds: float) -> int:
    return max(0, int(round(seconds * 1000.0)))


def _box(r: Rect) -> Box:
    return Box(
        x=round(r.x, 1), y=round(r.y, 1), w=round(max(r.w, 0.0), 1), h=round(max(r.h, 0.0), 1)
    )


def _clean(v: float, nd: int) -> float:
    r = round(float(v), nd)
    return 0.0 if r == 0 else r  # no "-0.0"


def ir_value(v: Value) -> Value:
    """Round numeric IR values for output (px 2 dp, ratio 4 dp); others unchanged."""
    if isinstance(v, PxValue):
        return PxValue(number=_clean(v.number, 2))
    if isinstance(v, RatioValue):
        return RatioValue(number=_clean(v.number, 4))
    return v


def _delta(a: Value, b: Value) -> float | None:
    if isinstance(a, PxValue | RatioValue) and isinstance(b, PxValue | RatioValue):
        return _clean(b.number - a.number, 6)
    return None


def warning(code: str, message: str | None = None, severity: str | None = None) -> SpecWarning:
    """Build a warning with the default text for ``code`` when no message is given."""
    sev, text = _WARNING_TEXT.get(code, ("info", code.replace("_", " ")))
    return SpecWarning(code=code, severity=severity or sev, message=message or text)  # type: ignore[arg-type]


def dedupe_warnings(warnings: Iterable[SpecWarning]) -> list[SpecWarning]:
    """First warning per code wins (stages may emit the same code twice)."""
    seen: dict[str, SpecWarning] = {}
    for w in warnings:
        seen.setdefault(w.code, w)
    return list(seen.values())


def transition_notes(raw: Iterable[str]) -> list[str]:
    out: list[str] = []
    for n in raw:
        mapped = _NOTE_MAP.get(n, n)
        if mapped and mapped not in out:
            out.append(mapped)
    return out


def ir_transition(ft: FittedTransition, onset_s: float, resolution_ms: int = 0) -> Transition:
    """``FittedTransition`` → IR ``Transition`` (times in int ms, values rounded).

    A delay shorter than one frame (``resolution_ms``) is below what the recording can resolve
    and is reported as 0 (``start_ms`` keeps the fitted value).
    """
    s = ft.series
    frm, to = ir_value(s.from_value), ir_value(s.to_value)
    start_ms = _ms(ft.t0_s)
    delay_ms = max(0, start_ms - _ms(onset_s))
    if delay_ms < resolution_ms:
        delay_ms = 0
    try:
        prog = normalize_progress(s.values, s.v_start, s.v_end)
        samples = progress_samples(s.times, prog)
    except ValueError:
        samples = []
    return Transition(
        id=ft.id,
        segment_id=s.segment,
        element_id=s.element_id,
        property=s.property,
        **{"from": frm},
        to=to,
        delta=_delta(frm, to),
        start_ms=start_ms,
        delay_ms=delay_ms,
        duration_ms=max(1, int(round(ft.duration_s * 1000.0))),
        easing=ft.easing,
        transform_origin=s.transform_origin if s.property in _SCALE_PROPS else None,
        confidence=ft.confidence,
        notes=transition_notes(s.notes),
        samples=samples,
    )


# --------------------------------------------------------------------------------------------
# Interpreter input
# --------------------------------------------------------------------------------------------


def _fmt_num(v: float) -> str:
    return f"{abs(v):.0f}" if abs(v) >= 3 else f"{abs(v):.1f}"


def change_summary(el: ElementCandidate, transitions: Sequence[FittedTransition]) -> str:
    """Forward changes of one element in words, e.g. ``moves up ~8px; scales up ~6%``."""
    parts: list[str] = []
    if el.kind == "appear":
        parts.append("appears")
    elif el.kind == "disappear":
        parts.append("disappears")
    elif el.kind == "backdrop":
        parts.append("page-wide overlay fades in")
    elif el.kind == "content_change":
        parts.append("content changes")
    for ft in transitions:
        s = ft.series
        if s.element_id != el.id or s.segment not in FORWARD_SEGMENTS:
            continue
        f, t = s.from_value, s.to_value
        d = _delta(f, t)
        if s.property == "translateY" and d is not None:
            parts.append(f"moves {'up' if d < 0 else 'down'} ~{_fmt_num(d)}px")
        elif s.property == "translateX" and d is not None:
            parts.append(f"moves {'left' if d < 0 else 'right'} ~{_fmt_num(d)}px")
        elif s.property in ("scale", "scaleX", "scaleY") and d is not None:
            pct = abs(d) * 100
            parts.append(f"{s.property} {'down' if d < 0 else 'up'} ~{pct:.0f}%")
        elif s.property == "opacity" and isinstance(f, RatioValue) and isinstance(t, RatioValue):
            parts.append(f"opacity {f.number:.2f} -> {t.number:.2f}")
        elif s.property in ("color", "background-color") and isinstance(f, ColorValue):
            parts.append(f"{s.property} {f.color} -> {t.color}")  # type: ignore[union-attr]
        elif s.property == "height" and d is not None:
            parts.append(f"height {'grows' if d > 0 else 'shrinks'} ~{_fmt_num(d)}px")
        elif s.property == "box-shadow":
            parts.append("shadow changes")
        elif s.property == "border-radius" and d is not None:
            parts.append(f"corner radius changes ~{_fmt_num(d)}px")
        elif s.property == "content":
            parts.append("content changes")
    parts = list(dict.fromkeys(parts))
    return "; ".join(parts) or "changes"


def interpretation_input(
    m: Measurement, job_dir: Path, keyframe_paths: dict[str, Path]
) -> InterpretationInput:
    """What Layer B may see (no numbers it could echo back as measurements)."""
    elements = [
        ElementSummary(
            id=e.id,
            bbox=_box(e.bbox_a if e.bbox_a.area > 0 else e.bbox_b),
            kind=e.kind,
            parent_id=e.parent_id,
            text_like=e.text_like,
            change_summary=change_summary(e, m.transitions),
        )
        for e in m.elements
    ]
    p = m.probe
    return InterpretationInput(
        job_dir=job_dir,
        keyframe_paths=keyframe_paths,
        elements=elements,
        heuristic_type=m.heuristic.type,
        heuristic_trigger=m.heuristic.trigger,
        cursor_summary=m.cursor_summary,
        video_meta=VideoMeta(
            width=p.width,
            height=p.height,
            duration_ms=max(1, _ms(p.duration_s)),
            pixel_ratio=m.scale.pixel_ratio,
        ),
    )


# --------------------------------------------------------------------------------------------
# Measured appearance (PLAN-continuous §13)
# --------------------------------------------------------------------------------------------

#: Appearance estimates with a raw score below this are too unreliable to hand to the coding
#: LLM and are dropped (absent = not measured).
APPEARANCE_MIN_SCORE = 0.15


def _measured_color(
    est: ColorEstimate | None, prop: str, params: MeasureParams
) -> MeasuredColor | None:
    if est is None or est.score < APPEARANCE_MIN_SCORE:
        return None
    conf = static_confidence(est.score, prop, params.confidence)
    return MeasuredColor(value=est.hex, confidence=conf)


def _measured_font_size(est: FontSizeEstimate | None) -> MeasuredNumber | None:
    if est is None or est.score < APPEARANCE_MIN_SCORE or est.px <= 0:
        return None
    return MeasuredNumber(
        value=round(est.px, 1), confidence=make_confidence(est.score, cap=FONT_SIZE_CONF_CAP)
    )


def appearance_statics(
    statics: Mapping[str, ElementStatic],
    appearance: AppearanceResult,
    params: MeasureParams = DEFAULT_PARAMS,
) -> dict[str, ElementStatic]:
    """``statics`` with the measured appearance merged in (new dict; inputs untouched).

    Colour confidences use the static colour cap (``cap_color``, 0.85); font size is capped at
    ``FONT_SIZE_CONF_CAP`` (medium at best). Radius / shadow already present are kept.
    """
    out = dict(statics)
    for eid, ea in appearance.elements.items():
        bg = _measured_color(ea.background, "background-color", params)
        fg = _measured_color(ea.text, "color", params)
        fs = _measured_font_size(ea.font_size)
        if bg is None and fg is None and fs is None:
            continue
        base = out.get(eid, ElementStatic())
        out[eid] = ElementStatic(
            border_radius_px=base.border_radius_px,
            shadow=base.shadow,
            background_color=bg,
            text_color=fg,
            font_size_px=fs,
        )
    return out


def appearance_scene(appearance: AppearanceResult, params: MeasureParams = DEFAULT_PARAMS) -> Scene:
    """IR ``Scene``: recorded viewport (CSS px) + page background."""
    w, h = appearance.viewport_css
    return Scene(
        viewport_css=Size(w=round(w, 1), h=round(h, 1)),
        page_background=_measured_color(appearance.page_background, "background-color", params),
    )


def apply_appearance(
    m: Measurement, appearance: AppearanceResult, params: MeasureParams = DEFAULT_PARAMS
) -> None:
    """Merge measured appearance into ``m`` (``statics`` + ``scene``), in place.

    P2 wiring (``run.measure``, after the statics block)::

        app = measure_appearance(AppearanceImage(st.a, (crop_css.x, crop_css.y), scale.k,
                                                 st.valid),
                                 elements_from_candidates(rr.elements), (fw, fh),
                                 page_image=<full state-A frame, optional>)
        apply_appearance(m, app, params)
    """
    m.statics = appearance_statics(m.statics, appearance, params)
    m.scene = appearance_scene(appearance, params)


# --------------------------------------------------------------------------------------------
# Merge
# --------------------------------------------------------------------------------------------


def merge_type(
    h: HeuristicInteraction, interp: InterpretationResult
) -> tuple[InteractionType, str, float, bool]:
    """``(type, type_source, confidence, disagrees)`` per §7.5."""
    if interp.status != "ok":
        return h.type, "heuristic", h.confidence, False
    it = interp.interaction
    agree = it.type_confirmation == h.type
    if it.type_confidence >= INTERPRETER_TYPE_MIN_CONF:
        if agree:
            # independent confirmation: keep the source, raise confidence (≤ 0.95 cap)
            return h.type, "heuristic", min(max(h.confidence, it.type_confidence), 0.95), False
        if h.confidence < HEURISTIC_CONFIDENT:
            return it.type_confirmation, "interpreter", min(it.type_confidence, 0.95), False
        return h.type, "heuristic", h.confidence, True
    return h.type, "heuristic", h.confidence, False


def _union(e: ElementCandidate) -> Rect:
    return e.bbox_a.union(e.bbox_b) if e.bbox_a.area > 0 else e.bbox_b


def merge_parents(
    elements: Sequence[ElementCandidate],
    interp: InterpretationResult,
    transitions: Sequence[FittedTransition],
) -> dict[str, str | None]:
    """Final parent per element id (§7.5 + the moving-parent guard in the module docstring)."""
    by_id = {e.id: e for e in elements}
    # A page overlay is a sibling of what opens on top of it, never its CSS parent: nesting the
    # panel inside a 50 %-opacity backdrop would fade the panel too. (CV keeps the nesting for
    # measurement: the panel is decomposed against the dimmed page; its warp is relative to the
    # backdrop's identity warp, so its values are already absolute.)
    parents = {
        e.id: None if e.parent_id in by_id and by_id[e.parent_id].kind == "backdrop"
        else e.parent_id
        for e in elements
    }  # fmt: skip
    if interp.status != "ok":
        return parents
    moving = {ft.series.element_id for ft in transitions if ft.series.property in GEOMETRIC_PROPS}
    for e in elements:
        ie = interp.elements.get(e.id)
        if ie is None or ie.parent_id == parents[e.id]:
            continue
        new = ie.parent_id
        if new is not None and (new not in by_id or by_id[new].kind == "backdrop"):
            continue
        if e.parent_id in moving or new in moving:
            continue
        if new is not None and _union(e).contained_fraction(_union(by_id[new])) < (
            REPARENT_MIN_CONTAINMENT
        ):
            continue
        parents[e.id] = new
    # never create a cycle
    for eid in list(parents):
        seen = {eid}
        p = parents[eid]
        while p is not None:
            if p in seen:
                parents[eid] = None
                break
            seen.add(p)
            p = parents.get(p)
    return parents


def _trigger_description(kind: str, target_label: str, cursor_visible: bool) -> str:
    target = target_label or "the target"
    if kind == "pointer_enter":
        return f"Pointer enters {target}"
    if kind == "click":
        return f"Click (cursor stops, then {target} changes)"
    if not cursor_visible:
        return "Trigger could not be determined (no cursor visible)"
    return "Trigger could not be determined"


def assemble(
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
    """Build the validated IR (see module docstring for the merge policy).

    Continuous measurements (``m.continuous``) are assembled by
    ``continuous.assemble.assemble_continuous`` (PLAN-continuous §4.7).
    """
    if m.continuous is not None:
        from app.pipeline.continuous.assemble import assemble_continuous

        return assemble_continuous(
            m, interp, job_id=job_id, filename=filename, generated_at=generated_at,
            pipeline_version=pipeline_version, use_interpreter=use_interpreter,
            extra_warnings=extra_warnings, params=params,
        )  # fmt: skip
    p = m.probe
    parents = merge_parents(m.elements, interp, m.transitions)

    # ---- elements (keep those with transitions + their ancestors) ---------------------------
    with_t = {ft.series.element_id for ft in m.transitions}
    keep: set[str] = set()
    for eid in with_t:
        cur: str | None = eid
        while cur is not None and cur not in keep:
            keep.add(cur)
            cur = parents.get(cur)
    elements: list[MotionElement] = []
    for e in m.elements:
        if e.id not in keep:
            continue
        ie = interp.elements.get(e.id)
        ok = interp.status == "ok" and ie is not None
        elements.append(
            MotionElement(
                id=e.id,
                label=ie.label if ie else f"Element ({e.id})",
                label_source=ie.label_source if ok else "heuristic",  # type: ignore[union-attr]
                role=ie.role if ie else "other",
                parent_id=parents.get(e.id) if parents.get(e.id) in keep else None,
                kind=e.kind,
                bbox_initial=_box(e.bbox_a),
                bbox_active=_box(e.bbox_b),
                text_like=e.text_like,
                static=m.statics.get(e.id, ElementStatic()),
            )
        )
    label_of = {e.id: e.label for e in elements}
    role_of = {e.id: e.role for e in elements}

    # ---- segments + transitions --------------------------------------------------------------
    by_seg: dict[str, list[FittedTransition]] = {}
    for ft in m.transitions:
        by_seg.setdefault(ft.series.segment, []).append(ft)
    segments: list[Segment] = []
    transitions: list[Transition] = []
    onset_of: dict[str, float] = {}
    for sw in sorted(m.segments, key=lambda s: SEGMENT_ORDER.index(s.id)):
        fts = by_seg.get(sw.id, [])
        if fts:
            onset = min(f.t0_s for f in fts)
            settle = max(f.t0_s + f.duration_s for f in fts)
        else:
            onset, settle = sw.start_s, sw.end_s
        onset_of[sw.id] = onset
        start_ms, end_ms = _ms(sw.start_s), _ms(sw.end_s)
        on_ms, set_ms = _ms(onset), _ms(settle)
        segments.append(
            Segment(
                id=sw.id,
                kind="forward" if sw.id in FORWARD_SEGMENTS else "reverse",
                start_ms=start_ms,
                end_ms=max(end_ms, start_ms),
                onset_ms=on_ms,
                settle_ms=max(set_ms, on_ms),
                trigger_event_ms=None if sw.trigger_event_s is None else _ms(sw.trigger_event_s),
            )
        )
    for ft in m.transitions:
        transitions.append(
            ir_transition(ft, onset_of.get(ft.series.segment, ft.t0_s), m.timing_resolution_ms)
        )

    seg_by_id = {s.id: s for s in segments}
    fwd = seg_by_id.get("fwd") or seg_by_id.get("rt_in")
    rev = seg_by_id.get("rev") or seg_by_id.get("rt_out")
    assert fwd is not None

    # ---- interaction -------------------------------------------------------------------------
    h = m.heuristic
    itype, type_source, type_conf, disagrees = merge_type(h, interp)
    target = m.target_id
    if interp.status == "ok" and interp.interaction.target_element_id in label_of:
        target = interp.interaction.target_element_id
    if target not in label_of:
        target = None
    target_label = label_of.get(target, "") if target else ""
    trigger_desc = _trigger_description(h.trigger, target_label, m.cursor_visible)
    if interp.status == "ok" and interp.interaction.trigger_description:
        trigger_desc = interp.interaction.trigger_description
    target_desc = interp.interaction.target_description if interp.status == "ok" else None
    interaction = Interaction(
        type=itype,
        type_source=type_source,  # type: ignore[arg-type]
        type_confidence=make_confidence(type_conf),
        pattern=pattern_for(role_of.get(target) if target else None, itype),
        target_element_id=target,
        target_label=target_label or "the changed element",
        target_description=target_desc,
        trigger=Trigger(
            kind=h.trigger,
            reverse_kind=h.reverse_trigger,
            description=trigger_desc,
            confidence=make_confidence(h.confidence),
        ),
        direction=m.direction,  # type: ignore[arg-type]
        total_duration_ms=TotalDuration(
            forward=fwd.settle_ms - fwd.onset_ms,
            reverse=None if rev is None else rev.settle_ms - rev.onset_ms,
        ),
    )

    # ---- relationships (need the final target + roles) ---------------------------------------
    area = {e.id: max(e.bbox_active.w * e.bbox_active.h, e.bbox_initial.w * e.bbox_initial.h)
            for e in elements}  # fmt: skip
    timed = [
        TimedTransition(
            id=ft.id,
            segment=ft.series.segment,
            element_id=ft.series.element_id,
            property=ft.series.property,
            t0_s=ft.t0_s,
            duration_s=ft.duration_s,
            area=area.get(ft.series.element_id),
            role=role_of.get(ft.series.element_id),
        )
        for ft in m.transitions
    ]
    rels = relationships(
        timed,
        target_element_id=target,
        timing_resolution_s=m.timing_resolution_ms / 1000.0,
        params=params.relationships,
    )

    # ---- warnings ----------------------------------------------------------------------------
    warns = list(m.warnings) + list(extra_warnings)
    if not m.cursor_visible:
        warns.append(warning("cursor_not_visible"))
    if m.direction == "forward":
        warns.append(warning("reverse_not_recorded"))
    if use_interpreter and interp.status in ("fallback", "timeout", "error"):
        reason = {"timeout": "timed out", "error": "failed", "fallback": "was unavailable"}
        warns.append(
            warning(
                "interpretation_fallback",
                f"AI labeling {reason[interp.status]}; element names and roles are heuristic.",
            )
        )
    if disagrees:
        warns.append(
            warning(
                "interpretation_disagrees",
                f"AI labeling suggested '{interp.interaction.type_confirmation}', but the measured "
                f"cursor and motion evidence points to '{h.type}'; the measured type is kept.",
                "info",
            )
        )

    cursor_events = [
        ev if ev.element_id is None or ev.element_id in label_of
        else CursorEvent(t_ms=ev.t_ms, kind=ev.kind, element_id=None)
        for ev in m.cursor_events
    ]  # fmt: skip

    return MotionSpec(
        job_id=job_id,
        meta=Meta(generated_at=generated_at, pipeline_version=pipeline_version),
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
        scene=m.scene,
        interaction=interaction,
        elements=elements,
        segments=segments,
        transitions=transitions,
        relationships=rels,
        cursor=Cursor(
            visible=m.cursor_visible,
            confidence=round(min(max(m.cursor_confidence, 0.0), 1.0), 3),
            events=cursor_events,
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
    "APPEARANCE_MIN_SCORE",
    "Measurement",
    "SegmentWindow",
    "appearance_scene",
    "appearance_statics",
    "apply_appearance",
    "assemble",
    "change_summary",
    "dedupe_warnings",
    "interpretation_input",
    "ir_transition",
    "merge_parents",
    "merge_type",
    "warning",
]
