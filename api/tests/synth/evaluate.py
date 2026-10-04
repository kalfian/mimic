"""Compare a pipeline result (IR) against synthetic ground truth (PLAN §11.4).

* :func:`compare` — truth + MotionSpec (or a pipeline error code) → :class:`Report` with one
  row per truth transition plus interaction / relationship / error checks.
* :func:`oracle_spec` — the truth expressed as a valid ``MotionSpec`` (perfect pipeline). Used
  to self-test the evaluator and to prove the truth format fits the frozen IR contract.

Easing curves are evaluated with the suite's independent :class:`synth.animate.CubicBezier`
(never ``app.pipeline.easing``). Element ids are matched by bounding-box IoU.

PLAN-continuous additions:

* :func:`compare_continuous` (called by :func:`compare` for ``truth.mode == "continuous"``):
  mode / axis / autoplay direction, speed, loop period (no false period), pitch, phase sequence
  (consecutive same-kind phases merged) and per-phase boundaries / durations / τ / v0, and the
  behaviour values (pause trigger, drag peak + pointer ratio, snap grid, resume delay / ramp)
  against the §8.4 targets (:data:`CONTINUOUS_THRESHOLDS`).
* :func:`oracle_spec` also builds a valid continuous IR from a continuous truth, and puts the
  appearance truth (§13) into ``scene`` / ``ElementStatic`` of every oracle.
* Appearance values (§13) are judged when the pipeline reports them (:data:`APPEARANCE`: fill
  ΔE ≤ 5, text ΔE ≤ 10, card box ±2 px; absent = not measured, not judged); font size, radius
  and the region box are informational. Exception (P2b, §14): a continuous scroller's pitch,
  gap and card box are part of the replica's CSS, so their absence fails.
* Card scale (P2c): a zoomed truth (C13) needs ``continuous.card_scale`` within
  ``card_scale_abs`` at the IR's reference distance and ``card_scale_mean_abs`` on the mean
  magnification; a rigid truth must not report one.
* ``drag.pointer_ratio`` is not judged when the IR reports it as unknown (the pointer is hidden
  inside a scroller until P4 in-band detection); a reported wrong ratio still fails.
* Pending gate: truths with ``suite == "continuous"`` are reported as *pending* (not run, not
  judged) until :data:`CONTINUOUS_SUITE_ENABLED` is switched on in P2.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import cv2
import numpy as np

from app.models.ir import (
    Autoplay,
    Behavior,
    Box,
    CardScale,
    Confidence,
    ConstantFit,
    ContinuousMotion,
    Cursor,
    CursorEvent,
    DragBehavior,
    Easing,
    ElementStatic,
    ExponentialFit,
    InertiaBehavior,
    Interaction,
    Interpretation,
    LoopInfo,
    MeasuredColor,
    MeasuredNumber,
    Meta,
    MotionElement,
    MotionSpec,
    PauseBehavior,
    Phase,
    RampFit,
    Relationship,
    ResumeBehavior,
    Scene,
    Segment,
    Size,
    SnapBehavior,
    Source,
    Span,
    SpecWarning,
    TotalDuration,
    Transition,
    TransitionConfidence,
    Trigger,
    TweenFit,
)

from .animate import CubicBezier
from .truth import Truth, TruthContinuous, TruthEasing, TruthPhase, TruthTransition

#: PLAN-continuous scenarios (``truth.suite == "continuous"``: C1–C12, N1) were reported as
#: *pending* until the pipeline analysed continuous motion; switched on in P2.
CONTINUOUS_SUITE_ENABLED = True


def is_pending(truth: Truth) -> bool:
    """True for scenarios the pipeline is not expected to handle yet (not run, not judged)."""
    return truth.suite == "continuous" and not CONTINUOUS_SUITE_ENABLED


#: PLAN §11.4 thresholds. "fast" = 60 fps CFR; "slow" = 30 fps or VFR.
THRESHOLDS: dict[str, Any] = {
    "translate_px": 0.5,  # applies when |delta| >= 4 px
    "translate_min_delta": 4.0,
    "scale": 0.01,
    "duration": {"fast": (20.0, 0.10), "slow": (35.0, 0.12)},  # max(abs ms, rel)
    "onset_ms": {"fast": 17.0, "slow": 34.0},
    "family_min_duration_ms": 150,
    "progress_rmse": 0.05,
    "delta_e_fill": 5.0,
    "delta_e_text": 10.0,
    "opacity": 0.1,
    "stagger_interval_ms": 15.0,
    "element_iou": 0.3,
}

#: Measured appearance (PLAN-continuous §13, set in P2): a reported colour must be within these
#: ΔE76 (fills as §11.4 colour transitions; thin text strokes lose chroma to H.264 4:2:0), a
#: reported card box within ``box_px``. Font size / radius stay informational.
APPEARANCE: dict[str, float] = {"fill_de": 5.0, "text_de": 10.0, "box_px": 2.0}

#: Family confusions not penalized (PLAN Appendix B note).
FAMILY_EQUIV = {frozenset({"ease", "ease-in-out"})}

#: PLAN-continuous §8.4 targets. "fast" = 60 fps CFR CRF 18; "slow" = 30 fps, VFR or CRF >= 28.
CONTINUOUS_THRESHOLDS: dict[str, Any] = {
    "speed_rel": {"fast": 0.02, "slow": 0.03},  # autoplay speed
    "loop_rel": 0.01,  # loop period (observable only; never a false period)
    "pitch_px": 2.0,  # pitch / snap step
    "tau_rel": {"fast": 0.15, "slow": 0.25},
    "tau_min_ms": 120.0,  # τ judged only when the truth τ >= this
    "v0_rel": {"fast": 0.15, "slow": 0.25},  # release velocity
    "peak_rel": {"fast": 0.10, "slow": 0.15},  # drag peak speed
    "sharp_ms": {"fast": 34.0, "slow": 67.0},  # drag start, release, snap/stop end
    "soft": {"fast": (60.0, 0.15), "slow": (90.0, 0.20)},  # decel/resume t0: max(ms, rel*ramp)
    "ramp": {"fast": (60.0, 0.20), "slow": (90.0, 0.25)},  # ramp/decel/snap duration
    "resume_delay_ms": {"fast": 50.0, "slow": 80.0},
    "pointer_ratio": 0.15,  # |ratio - 1| (PLAN-continuous §4.5 step 8)
    "region_iou": 0.3,
    # P2c: card scale at the IR's reference distance vs the truth's lens at that distance, and
    # the mean magnification (the speed conversion; the centre is ×1 by definition and its card
    # size is judged by the card box, ±2 px)
    "card_scale_abs": 0.03,
    "card_scale_mean_abs": 0.02,
}


@dataclass
class Row:
    segment: str
    element: str
    ir_element: str | None
    prop: str
    truth: str
    measured: str
    value_err: str
    value_ok: bool | None
    start_err_ms: float | None
    start_ok: bool | None
    dur_err_ms: float | None
    dur_ok: bool | None
    family: str
    family_ok: bool | None
    rmse: float | None
    rmse_ok: bool | None
    note: str = ""
    band: str | None = None  # IR confidence band of the matched transition
    timing_conf: float | None = None
    eligible_family: bool = False  # counts toward the aggregate family metric (§11.4)

    @property
    def ok(self) -> bool:
        """Per-transition thresholds. Easing family is an *aggregate* §11.4 metric (≥ 80 % of
        eligible transitions), so it is reported per row but judged in :func:`summarize`."""
        checks = (self.value_ok, self.start_ok, self.dur_ok, self.rmse_ok)
        return self.ir_element is not None and all(c is not False for c in checks)

    @property
    def within_targets(self) -> bool:
        """Everything this row measures is within target, including the family when eligible."""
        return self.ok and self.family_ok is not False


@dataclass
class Report:
    name: str
    rows: list[Row] = field(default_factory=list)
    suite: str = "transition"
    checks: list[tuple[str, bool | None, str]] = field(default_factory=list)  # (what, ok, detail)
    extra: list[str] = field(default_factory=list)
    #: continuous values reported with a confidence: (what, band, within target)
    cvalues: list[tuple[str, str, bool]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(r.ok for r in self.rows) and all(c[1] is not False for c in self.checks)


# --------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------


def _iou(a, b) -> float:
    x0, y0 = max(a.x, b.x), max(a.y, b.y)
    x1, y1 = min(a.x + a.w, b.x + b.w), min(a.y + a.h, b.y + b.h)
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union > 0 else 0.0


def match_elements(truth: Truth, spec: MotionSpec) -> dict[str, str]:
    """truth element id -> IR element id (greedy by IoU, each IR element used once)."""
    pairs = []
    for te in truth.elements:
        tbox = te.bbox_initial if te.visible_initial else te.bbox_active
        for ie in spec.elements:
            iou = max(_iou(tbox, ie.bbox_initial), _iou(tbox, ie.bbox_active))
            if iou >= THRESHOLDS["element_iou"]:
                pairs.append((iou, te.id, ie.id))
    out: dict[str, str] = {}
    used: set[str] = set()
    for _, tid, iid in sorted(pairs, reverse=True):
        if tid not in out and iid not in used:
            out[tid] = iid
            used.add(iid)
    return out


def hex_lab(h: str) -> np.ndarray:
    rgb = np.array([[[int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)]]], np.float32) / 255.0
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab)[0, 0].astype(np.float64)


def delta_e76(a: str, b: str) -> float:
    return float(np.linalg.norm(hex_lab(a) - hex_lab(b)))


def _fmt_value(v) -> str:
    if v.kind in ("px", "ratio"):
        return f"{v.number:g}"
    if v.kind == "color":
        return v.color
    if v.kind == "shadow":
        s = v.shadow
        return "none" if s is None else f"{s.x:g} {s.y:g} {s.blur:g} a{s.rgba[3]:g}"
    return v.text


def progress_rmse(
    truth_t: TruthTransition, start_ms: float, dur_ms: float, bezier: tuple[float, ...]
) -> float:
    """RMSE between the truth progress curve and a fitted curve over the union of both spans."""
    t0 = min(truth_t.start_ms, start_ms) - 50.0
    t1 = max(truth_t.start_ms + truth_t.duration_ms, start_ms + dur_ms) + 50.0
    t = np.arange(t0, t1 + 0.5, 1.0)
    pt = CubicBezier(*truth_t.easing.cubic_bezier)(
        np.clip((t - truth_t.start_ms) / truth_t.duration_ms, 0, 1)
    )
    pf = CubicBezier(*bezier)(np.clip((t - start_ms) / dur_ms, 0, 1))
    return float(np.sqrt(np.mean((pt - pf) ** 2)))


def _speed(truth: Truth) -> str:
    return "slow" if truth.video.vfr or truth.video.fps < 50 else "fast"


def _cspeed(truth: Truth) -> str:
    """Continuous targets are relaxed for 30 fps, VFR and CRF >= 28 (§8.4 parentheses)."""
    v = truth.video
    return "slow" if v.vfr or v.fps < 50 or v.crf >= 28 else "fast"


# --------------------------------------------------------------------------------------------
# Compare
# --------------------------------------------------------------------------------------------


def compare(truth: Truth, spec: MotionSpec | None, error_code: str | None = None) -> Report:
    rep = Report(name=truth.name, suite=truth.suite)
    if truth.expected_error is not None:
        rep.checks.append(
            (
                "expected_error",
                error_code == truth.expected_error,
                f"want {truth.expected_error}, got {error_code or 'a result'}",
            )  # fmt: skip
        )
        return rep
    if spec is None:
        rep.checks.append(("result", False, f"pipeline failed: {error_code}"))
        return rep
    if truth.continuous is not None:
        return compare_continuous(truth, spec, rep)
    if truth.suite == "continuous" or spec.mode != truth.mode:
        rep.checks.append(("mode", spec.mode == truth.mode, f"want {truth.mode}, got {spec.mode}"))

    speed = _speed(truth)
    eid = match_elements(truth, spec)
    for te in truth.elements:
        if te.id not in eid:
            rep.checks.append((f"element {te.id}", False, "no IR element with IoU >= 0.3"))

    used_ir: set[str] = set()
    for tt in truth.transitions:
        iel = eid.get(tt.element_id)
        it: Transition | None = None
        if iel is not None:
            it = next(
                (x for x in spec.transitions
                 if x.element_id == iel and x.property == tt.property
                 and x.segment_id == tt.segment_id),
                None,
            )  # fmt: skip
        if it is None:
            rep.rows.append(
                Row(
                    tt.segment_id,
                    tt.element_id,
                    None,
                    tt.property,
                    f"{_fmt_value(tt.from_)}→{_fmt_value(tt.to)}",
                    "MISSING",
                    "",
                    None,
                    None,
                    None,
                    None,
                    None,
                    tt.easing.family,
                    None,
                    None,
                    None,
                    note="missing transition",
                )  # fmt: skip
            )
            continue
        used_ir.add(it.id)
        rep.rows.append(_row(truth, tt, it, iel, speed))  # type: ignore[arg-type]

    for x in spec.transitions:
        if x.id not in used_ir:
            rep.extra.append(f"{x.id} {x.segment_id} {x.element_id} {x.property}")

    inter = truth.interaction
    assert inter is not None
    strict_type = truth.cursor.visible
    rep.checks.append(
        (
            "interaction.type",
            (spec.interaction.type in inter.accept_types) if strict_type else None,
            f"want {'|'.join(inter.accept_types)}, got {spec.interaction.type}",
        )  # fmt: skip
    )
    rep.checks.append(
        ("interaction.trigger", None, f"want {inter.trigger}, got {spec.interaction.trigger.kind}")
    )
    rep.checks.append(
        (
            "interaction.direction",
            spec.interaction.direction == inter.direction,
            f"want {inter.direction}, got {spec.interaction.direction}",
        )  # fmt: skip
    )
    _relationships(truth, spec, eid, rep)
    _expected_warnings(truth, spec, rep)
    _appearance_info(truth, spec, eid, rep)
    return rep


def _expected_warnings(truth: Truth, spec: MotionSpec, rep: Report) -> None:
    got = {w.code for w in spec.warnings}
    for code in truth.expected_warnings:
        rep.checks.append((f"warning {code}", code in got, f"got {sorted(got) or 'none'}"))


def _color_check(name: str, want: str | None, got: MeasuredColor | None, limit: float) -> tuple:
    """A reported colour must be within ``limit`` ΔE76; absent = not measured (not judged)."""
    if want is None or got is None:
        return (name, None, f"not reported (truth {want})")
    de = delta_e76(want, got.value)
    return (name, de <= limit, f"ΔE{de:.1f} (≤ {limit:g}; {got.confidence.band})")


def _scene_checks(truth: Truth, spec: MotionSpec, rep: Report) -> None:
    if truth.scene is None or spec.scene is None:
        return
    vp = spec.scene.viewport_css
    tv = truth.scene.viewport_css
    rep.checks.append(("scene.viewport", vp.w == tv.w and vp.h == tv.h,
                       f"truth {tv.w:g}x{tv.h:g}, got {vp.w:g}x{vp.h:g}"))  # fmt: skip
    if spec.scene.page_background is not None:
        want, got = truth.scene.page_background, spec.scene.page_background
        rep.checks.append(_color_check("scene.page_background", want, got, APPEARANCE["fill_de"]))


def _appearance_info(truth: Truth, spec: MotionSpec, eid: dict[str, str], rep: Report) -> None:
    """Appearance truth (§13) next to measured values. Colours are judged when reported
    (:data:`APPEARANCE`; absent = not measured, not judged); font size and radius are
    informational (approximate by design, confidence capped)."""
    _scene_checks(truth, spec, rep)
    by_id = {e.id: e for e in spec.elements}
    for te in truth.elements:
        ie = by_id.get(eid.get(te.id, ""))
        ap = te.appearance
        if ie is None or ap is None:
            continue
        st = ie.static
        for name, want, got, limit in (
            ("background", ap.background_color, st.background_color, APPEARANCE["fill_de"]),
            ("text_color", ap.text_color, st.text_color, APPEARANCE["text_de"]),
        ):
            if want is not None and got is not None:
                rep.checks.append(_color_check(f"{te.id}.{name}", want, got, limit))
        if ap.font_size_px is not None and st.font_size_px is not None:
            rep.checks.append(
                (f"{te.id}.font_size", None,
                 f"truth font {ap.font_size_px:g}px (cap height {ap.cap_height_px}px), "
                 f"got {st.font_size_px.value:g}px")
            )  # fmt: skip
        if ap.border_radius_px is not None and st.border_radius_px is not None:
            rep.checks.append(
                (f"{te.id}.border_radius", None,
                 f"truth {ap.border_radius_px:g}px, got {st.border_radius_px.value:g}px")
            )  # fmt: skip


# --------------------------------------------------------------------------------------------
# Continuous compare (PLAN-continuous §8.4)
# --------------------------------------------------------------------------------------------


@dataclass
class _Merged:
    """Consecutive IR phases of one kind (merged for the sequence check)."""

    kind: str
    start_ms: int
    end_ms: int
    phases: list[Phase]


def merge_phases(phases: list[Phase]) -> list[_Merged]:
    out: list[_Merged] = []
    for ph in phases:
        if out and out[-1].kind == ph.kind:
            out[-1].end_ms = ph.end_ms
            out[-1].phases.append(ph)
        else:
            out.append(_Merged(ph.kind, ph.start_ms, ph.end_ms, [ph]))
    return out


def merge_truth_phases(phases: list[TruthPhase]) -> list[TruthPhase]:
    out: list[TruthPhase] = []
    for ph in phases:
        if out and out[-1].kind == ph.kind:
            out[-1] = out[-1].model_copy(update={"end_ms": ph.end_ms})
        else:
            out.append(ph)
    return out


def _time_iou(a0: float, a1: float, b0: float, b1: float) -> float:
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 0 else 0.0


def match_phases(
    truth_phases: list[TruthPhase], ir: list[_Merged]
) -> list[tuple[int, TruthPhase, _Merged]]:
    """Pairs (truth index, truth phase, IR phase): in order when the kind sequences agree,
    otherwise greedily by maximum time-IoU among phases of the same kind."""
    if [p.kind for p in truth_phases] == [m.kind for m in ir]:
        return [(i, tp, m) for i, (tp, m) in enumerate(zip(truth_phases, ir, strict=True))]
    cands = sorted(
        (
            (_time_iou(tp.start_ms, tp.end_ms, m.start_ms, m.end_ms), i, j)
            for i, tp in enumerate(truth_phases)
            for j, m in enumerate(ir)
            if tp.kind == m.kind
        ),
        reverse=True,
    )
    used_t: set[int] = set()
    used_i: set[int] = set()
    pairs = []
    for iou, i, j in cands:
        if iou > 0 and i not in used_t and j not in used_i:
            used_t.add(i)
            used_i.add(j)
            pairs.append((i, truth_phases[i], ir[j]))
    return sorted(pairs, key=lambda x: x[0])


def _rel(got: float, want: float) -> float:
    return abs(got - want) / abs(want) if want else abs(got)


def compare_continuous(truth: Truth, spec: MotionSpec, rep: Report) -> Report:
    tc = truth.continuous
    assert tc is not None
    th, sp = CONTINUOUS_THRESHOLDS, _cspeed(truth)
    chk = rep.checks
    chk.append(("mode", spec.mode == "continuous", f"want continuous, got {spec.mode}"))
    c = spec.continuous
    if c is None:
        return rep
    inter = truth.interaction
    if inter is not None:
        chk.append((
            "interaction.type",
            spec.interaction.type in inter.accept_types,
            f"want {'|'.join(inter.accept_types)}, got {spec.interaction.type}",
        ))  # fmt: skip
    chk.append(("axis", c.axis == tc.axis, f"want {tc.axis}, got {c.axis}"))
    iou = _iou(tc.region, c.region)
    chk.append(("region", iou >= th["region_iou"], f"IoU {iou:.2f}"))

    # autoplay + loop
    v_auto = tc.autoplay_velocity_px_s
    if v_auto is None:
        chk.append(("autoplay", c.autoplay is None, "want none"))
    elif c.autoplay is None:
        chk.append(("autoplay", False, f"want {v_auto:+g} px/s, got none"))
    else:
        ap = c.autoplay
        chk.append(("autoplay.direction", ap.direction == tc.autoplay_direction,
                    f"want {tc.autoplay_direction}, got {ap.direction}"))  # fmt: skip
        err = _rel(ap.speed_px_s.value, abs(v_auto))
        chk.append(("autoplay.speed", err <= th["speed_rel"][sp],
                    f"want {abs(v_auto):g}, got {ap.speed_px_s.value:g} ({err:.1%})"))  # fmt: skip
        rep.cvalues.append(("autoplay.speed", ap.speed_px_s.confidence.band,
                            err <= th["speed_rel"][sp]))  # fmt: skip
        loop = ap.loop
        if tc.loop.observable:
            if not loop.observed or loop.period_px is None:
                chk.append(("loop.period", False, f"want {tc.loop.period_px:g} px, not observed"))
            else:
                err = _rel(loop.period_px.value, tc.loop.period_px)
                rep.cvalues.append(("loop.period", loop.period_px.confidence.band,
                                    err <= th["loop_rel"]))  # fmt: skip
                chk.append(("loop.period", err <= th["loop_rel"],
                            f"want {tc.loop.period_px:g}, got {loop.period_px.value:g} "
                            f"({err:.2%})"))  # fmt: skip
        else:
            got = loop.period_px.value if loop.period_px is not None else None
            chk.append(
                (
                    "loop.not_observed",
                    not loop.observed,
                    "no false period" if not loop.observed else f"false period {got}",
                )
            )
    if c.pitch_px is not None:
        d = abs(c.pitch_px.value - tc.pitch_px)
        chk.append(("pitch", d <= th["pitch_px"], f"want {tc.pitch_px:g}, got "
                    f"{c.pitch_px.value:g}"))  # fmt: skip
        rep.cvalues.append(("pitch", c.pitch_px.confidence.band, d <= th["pitch_px"]))
    else:
        # §14 (P2b): the card spacing is part of the replica's CSS, so a missing pitch / gap /
        # card fails (the motivating recording's original failure: no cards found at all)
        chk.append(("pitch", False, f"not reported (truth {tc.pitch_px:g})"))
    if c.gap_px is not None:
        d = abs(c.gap_px.value - tc.gap_px)
        chk.append(("gap", d <= th["pitch_px"], f"want {tc.gap_px:g}, got {c.gap_px.value:g}"))
        rep.cvalues.append(("gap", c.gap_px.confidence.band, d <= th["pitch_px"]))
    else:
        chk.append(("gap", False, f"not reported (truth {tc.gap_px:g})"))
    _card_scale_checks(tc, c, chk, rep.cvalues)

    _continuous_phases(tc, c, sp, chk, rep.cvalues)
    _continuous_behavior(tc, c, sp, chk, rep.cvalues)
    _expected_warnings(truth, spec, rep)
    _continuous_appearance(truth, spec, rep)
    return rep


def _card_scale_checks(tc: TruthContinuous, c: ContinuousMotion, chk: list, cvalues: list) -> None:
    """P2c: position-dependent card scale. Rigid truth -> none reported (a false scale would
    make the replica's cards grow); zoomed truth -> required, judged at the IR's own reference
    distance against the truth lens ``s = 1 + a d²`` (``a`` from the edge scale at half the
    viewport) and by the mean magnification (``TruthZoom.speed_scale``)."""
    th = CONTINUOUS_THRESHOLDS
    z, cs = tc.zoom, c.card_scale
    if z is None:
        got = "none" if cs is None else f"false scale ×{cs.scale_at_reference:g}"
        chk.append(("card_scale", cs is None, f"rigid cards, got {got}"))
        return
    if cs is None:
        chk.append(("card_scale", False, f"not reported (truth ×{z.edge_scale:g} at the edges)"))
        return
    half = (tc.region.w if tc.axis == "x" else tc.region.h) / 2.0
    a = (z.edge_scale - 1.0) / (half * half)
    ref = cs.reference_distance_px
    want = 1.0 + a * ref * ref
    ok = abs(cs.scale_at_reference - want) <= th["card_scale_abs"]
    chk.append(("card_scale.at_ref", ok, f"want ×{want:.3f} at {ref:g} px, got "
                f"×{cs.scale_at_reference:.3f}"))  # fmt: skip
    cvalues.append(("card_scale", cs.confidence.band, ok))
    ok = abs(cs.mean_scale - z.speed_scale) <= th["card_scale_mean_abs"]
    chk.append(("card_scale.mean", ok, f"want ×{z.speed_scale:.3f}, got ×{cs.mean_scale:.3f}"))


def _continuous_appearance(truth: Truth, spec: MotionSpec, rep: Report) -> None:
    """Scroller / card / title appearance (§13) of a continuous spec."""
    tc = truth.continuous
    c = spec.continuous
    assert tc is not None and c is not None
    chk = rep.checks
    _scene_checks(truth, spec, rep)
    r = c.region
    chk.append(("region.box", None, f"truth {tc.region.x:g},{tc.region.y:g} {tc.region.w:g}x"
                f"{tc.region.h:g}, got {r.x:g},{r.y:g} {r.w:g}x{r.h:g}"))  # fmt: skip
    scroller = next((e for e in spec.elements if e.id == c.element_id), None)
    t_scroller = next((e for e in truth.elements if e.id == tc.element_id), None)
    if scroller is not None and t_scroller is not None and t_scroller.appearance is not None:
        want = t_scroller.appearance.background_color
        got = scroller.static.background_color
        chk.append(_color_check("scroller.background", want, got, APPEARANCE["fill_de"]))
    card = next((e for e in spec.elements if e.role == "card" or (
        e.parent_id == c.element_id and not e.text_like)), None)  # fmt: skip
    if card is None:
        chk.append(("card", False, "no card element reported"))
        return
    b = card.bbox_initial
    tol = APPEARANCE["box_px"]
    ok = abs(b.w - tc.card.w) <= tol and abs(b.h - tc.card.h) <= tol
    chk.append(("card.size", ok, f"want {tc.card.w:g}x{tc.card.h:g}, got {b.w:g}x{b.h:g}"))
    chk.append(_color_check("card.background", tc.card.background_color,
                            card.static.background_color, APPEARANCE["fill_de"]))  # fmt: skip
    title = next((e for e in spec.elements if e.parent_id == card.id and e.text_like), None)
    if title is None:
        chk.append(("card.title", None, "no title text reported"))
        return
    chk.append(_color_check("card.title.text_color", tc.card.text_color,
                            title.static.text_color, APPEARANCE["text_de"]))  # fmt: skip
    fs = title.static.font_size_px
    chk.append(("card.title.font_size", None,
                f"truth {tc.card.font_size_px:g}px (cap {tc.card.cap_height_px:g}px), got "
                f"{fs.value:g}px" if fs is not None else "not reported"))  # fmt: skip


def _continuous_phases(
    tc: TruthContinuous, c: ContinuousMotion, sp: str, chk: list, cv: list
) -> None:
    th = CONTINUOUS_THRESHOLDS
    tph = merge_truth_phases(tc.phases)
    irm = merge_phases(c.phases)
    want = [p.kind for p in tph]
    got = [m.kind for m in irm]
    chk.append(
        (
            "phase sequence",
            want == got,
            f"want {' '.join(want)}" + ("" if want == got else f"; got {' '.join(got)}"),
        )
    )
    sharp = th["sharp_ms"][sp]
    soft_abs, soft_rel = th["soft"][sp]
    ramp_abs, ramp_rel = th["ramp"][sp]
    for i, tp, m in match_phases(tph, irm):
        tag = f"phase {i + 1} {tp.kind}"
        d_start, d_end = m.start_ms - tp.start_ms, m.end_ms - tp.end_ms
        if tp.kind == "drag":
            chk.append((f"{tag} start", abs(d_start) <= sharp, f"{d_start:+d} ms"))
            chk.append((f"{tag} release", abs(d_end) <= sharp, f"{d_end:+d} ms"))
        if tp.kind in ("snap", "stop"):
            chk.append((f"{tag} end", abs(d_end) <= sharp, f"{d_end:+d} ms"))
        if tp.kind in ("decelerate", "resume"):
            ramp = tp.ramp_ms or (tp.end_ms - tp.start_ms)
            tol = max(soft_abs, soft_rel * ramp)
            chk.append((f"{tag} t0", abs(d_start) <= tol, f"{d_start:+d} ms (tol {tol:.0f})"))
            fit = m.phases[0].fit
            dur = fit.duration_ms.value if isinstance(fit, RampFit) else m.end_ms - m.start_ms
            tol = max(ramp_abs, ramp_rel * ramp)
            chk.append((f"{tag} ramp", abs(dur - ramp) <= tol,
                        f"want {ramp} ms, got {dur:.0f} (tol {tol:.0f})"))  # fmt: skip
            if isinstance(fit, RampFit):
                cv.append((f"{tp.kind} ramp", fit.duration_ms.confidence.band,
                           abs(dur - ramp) <= tol))  # fmt: skip
        if tp.kind == "snap":
            want_d = tp.end_ms - tp.start_ms
            fit = m.phases[-1].fit
            dur = fit.duration_ms.value if isinstance(fit, TweenFit) else m.end_ms - m.start_ms
            tol = max(ramp_abs, ramp_rel * want_d)
            chk.append((f"{tag} duration", abs(dur - want_d) <= tol,
                        f"want {want_d} ms, got {dur:.0f} (tol {tol:.0f})"))  # fmt: skip
            if isinstance(fit, TweenFit):
                cv.append(("snap duration", fit.duration_ms.confidence.band,
                           abs(dur - want_d) <= tol))  # fmt: skip
        if tp.kind == "inertia":
            fit = m.phases[0].fit
            if not isinstance(fit, ExponentialFit):
                chk.append((f"{tag} model", False,
                            f"want exponential, got {fit.model if fit else 'none'}"))  # fmt: skip
                continue
            assert tp.tau_ms is not None and tp.v0_px_s is not None
            if tp.tau_ms >= th["tau_min_ms"]:
                err = _rel(fit.tau_ms.value, tp.tau_ms)
                cv.append(("inertia tau", fit.tau_ms.confidence.band, err <= th["tau_rel"][sp]))
                chk.append(
                    (
                        f"{tag} tau",
                        err <= th["tau_rel"][sp],
                        f"want {tp.tau_ms:g}, got {fit.tau_ms.value:g} ({err:.0%})",
                    )
                )
            err = _rel(fit.v0_px_s.value, tp.v0_px_s)
            cv.append(("inertia v0", fit.v0_px_s.confidence.band, err <= th["v0_rel"][sp]))
            chk.append(
                (
                    f"{tag} v0",
                    err <= th["v0_rel"][sp],
                    f"want {tp.v0_px_s:+g}, got {fit.v0_px_s.value:+g} ({err:.0%})",
                )
            )


def _continuous_behavior(
    tc: TruthContinuous, c: ContinuousMotion, sp: str, chk: list, cv: list
) -> None:
    th = CONTINUOUS_THRESHOLDS
    b = c.behavior
    ramp_abs, ramp_rel = th["ramp"][sp]
    # pause
    if tc.pause_on is None:
        chk.append(("pause", b.pause is None, "want none" if b.pause is None else "false pause"))
    elif b.pause is None:
        chk.append(("pause", False, f"want on {'|'.join(tc.pause_on_accept)}, got none"))
    else:
        chk.append(("pause.on", b.pause.on in tc.pause_on_accept,
                    f"want {'|'.join(tc.pause_on_accept)}, got {b.pause.on}"))  # fmt: skip
        if tc.pause_decel_ms is not None:
            got = b.pause.decel_ms.value if b.pause.decel_ms is not None else None
            tol = max(ramp_abs, ramp_rel * tc.pause_decel_ms)
            ok = got is not None and abs(got - tc.pause_decel_ms) <= tol
            chk.append(("pause.decel_ms", ok, f"want {tc.pause_decel_ms}, got {got}"))
            if b.pause.decel_ms is not None:
                cv.append(("pause.decel", b.pause.decel_ms.confidence.band, ok))
    # drag
    if tc.drag_count == 0:
        chk.append(("drag", b.drag is None, "want none" if b.drag is None else "false drag"))
    elif b.drag is None:
        chk.append(("drag", False, f"want {tc.drag_count} drags, got none"))
    else:
        assert tc.drag_peak_speed_px_s is not None
        err = _rel(b.drag.peak_speed_px_s.value, tc.drag_peak_speed_px_s)
        cv.append(("drag.peak", b.drag.peak_speed_px_s.confidence.band,
                   err <= th["peak_rel"][sp]))  # fmt: skip
        chk.append(("drag.peak_speed", err <= th["peak_rel"][sp],
                    f"want {tc.drag_peak_speed_px_s:g}, got {b.drag.peak_speed_px_s.value:g} "
                    f"({err:.0%})"))  # fmt: skip
        if tc.drag_follows_pointer:
            ratio = b.drag.pointer_ratio.value if b.drag.pointer_ratio is not None else None
            if b.drag.follows_pointer is None and ratio is None:
                # the pointer moves with the content during a drag, so it is hidden inside the
                # scroller until in-band pointer detection (PLAN-continuous P4); "not measured"
                # is reported honestly and not judged. A wrong measured value still fails.
                chk.append(("drag.pointer_ratio", None,
                            "not measured (pointer hidden inside the scroller; P4)"))  # fmt: skip
            else:
                ok = (
                    bool(b.drag.follows_pointer)
                    and ratio is not None
                    and (abs(ratio - (tc.drag_pointer_ratio or 1.0)) <= th["pointer_ratio"])
                )
                chk.append(
                    (
                        "drag.pointer_ratio",
                        ok,
                        f"want follows 1:1, got {b.drag.follows_pointer} ratio {ratio}",
                    )
                )
        else:
            chk.append(("drag.pointer_ratio", None, f"no cursor; got {b.drag.follows_pointer}"))
    # snap (no false snaps)
    if tc.snap_kind is None:
        chk.append(("snap", b.snap is None, "want none" if b.snap is None else
                    f"false snap ({b.snap.kind})"))  # fmt: skip
    elif b.snap is None or b.snap.kind != tc.snap_kind:
        chk.append(("snap", False, f"want {tc.snap_kind}, got {b.snap.kind if b.snap else None}"))
    else:
        step = b.snap.step_px.value if b.snap.step_px is not None else None
        ok = step is not None and abs(step - tc.pitch_px) <= th["pitch_px"]
        chk.append(("snap.step", ok, f"want {tc.pitch_px:g}, got {step}"))
        if b.snap.step_px is not None:
            cv.append(("snap.step", b.snap.step_px.confidence.band, ok))
    # resume
    if tc.resume_ramp_ms is None:
        chk.append(("resume", b.resume is None, "want none" if b.resume is None else
                    "false resume"))  # fmt: skip
    elif b.resume is None:
        chk.append(("resume", False, "want a resume, got none"))
    else:
        r = b.resume
        assert tc.resume_delay_after_rest_ms is not None
        d = r.delay_after_rest_ms.value - tc.resume_delay_after_rest_ms
        chk.append(("resume.delay_after_rest", abs(d) <= th["resume_delay_ms"][sp],
                    f"want {tc.resume_delay_after_rest_ms}, got "
                    f"{r.delay_after_rest_ms.value:.0f} ({d:+.0f} ms)"))  # fmt: skip
        cv.append(("resume.delay", r.delay_after_rest_ms.confidence.band,
                   abs(d) <= th["resume_delay_ms"][sp]))  # fmt: skip
        if tc.resume_delay_after_leave_ms is not None:
            got_l = r.delay_after_leave_ms
            if got_l is None:
                chk.append(("resume.delay_after_leave", False,
                            f"want {tc.resume_delay_after_leave_ms}, not reported"))  # fmt: skip
            else:
                d = got_l.value - tc.resume_delay_after_leave_ms
                cv.append(("resume.delay_after_leave", got_l.confidence.band,
                           abs(d) <= th["resume_delay_ms"][sp]))  # fmt: skip
                chk.append(("resume.delay_after_leave", abs(d) <= th["resume_delay_ms"][sp],
                            f"want {tc.resume_delay_after_leave_ms}, got {got_l.value:.0f} "
                            f"({d:+.0f} ms)"))  # fmt: skip
        tol = max(ramp_abs, ramp_rel * tc.resume_ramp_ms)
        d = r.ramp_ms.value - tc.resume_ramp_ms
        cv.append(("resume.ramp", r.ramp_ms.confidence.band, abs(d) <= tol))
        chk.append(
            (
                "resume.ramp",
                abs(d) <= tol,
                f"want {tc.resume_ramp_ms}, got {r.ramp_ms.value:.0f} (tol {tol:.0f})",
            )
        )
        chk.append(
            (
                "resume.direction_preserved",
                r.direction_preserved == tc.resume_direction_preserved,
                f"want {tc.resume_direction_preserved}, got {r.direction_preserved}",
            )
        )


def _row(truth: Truth, tt: TruthTransition, it: Transition, iel: str, speed: str) -> Row:
    th = THRESHOLDS
    kind = tt.from_.kind
    value_ok: bool | None = None
    value_err = ""
    if kind == "px":
        err = max(abs(it.from_.number - tt.from_.number), abs(it.to.number - tt.to.number))  # type: ignore[union-attr]
        value_err = f"{err:.2f}px"
        if (
            tt.property in ("translateX", "translateY")
            and abs(tt.delta or 0) >= th["translate_min_delta"]
        ):
            value_ok = err <= th["translate_px"]
    elif kind == "ratio":
        err = max(abs(it.from_.number - tt.from_.number), abs(it.to.number - tt.to.number))  # type: ignore[union-attr]
        value_err = f"{err:.3f}"
        limit = th["opacity"] if tt.property == "opacity" else th["scale"]
        value_ok = err <= limit
    elif kind == "color":
        de = max(delta_e76(it.from_.color, tt.from_.color), delta_e76(it.to.color, tt.to.color))  # type: ignore[union-attr]
        value_err = f"ΔE{de:.1f}"
        text_like = truth.element(tt.element_id).text_like
        value_ok = de <= (th["delta_e_text"] if text_like else th["delta_e_fill"])
    elif kind == "shadow":

        def strength(v) -> float:
            s = v.shadow
            return 0.0 if s is None else s.blur * s.rgba[3] + abs(s.y) * s.rgba[3]

        want = np.sign(strength(tt.to) - strength(tt.from_))
        got = np.sign(strength(it.to) - strength(it.from_))
        value_err = "dir ok" if want == got else "dir wrong"
        value_ok = bool(want == got)

    start_err = float(it.start_ms - tt.start_ms)
    start_ok: bool | None = abs(start_err) <= th["onset_ms"][speed]
    dur_err = float(it.duration_ms - tt.duration_ms)
    abs_ms, rel = th["duration"][speed]
    dur_ok: bool | None = abs(dur_err) <= max(abs_ms, rel * tt.duration_ms)
    # §11.4 judges box-shadow on direction only ("detected increases"): its progress signal is
    # ring darkening, a non-linear function of blur/offset/alpha, so its timing and curve are
    # reported (low confidence) but are not accuracy targets.
    timing_scored = kind != "shadow"
    if not timing_scored:
        start_ok = None if not start_ok else True
        dur_ok = None if not dur_ok else True

    fam_t, fam_m = tt.easing.family, it.easing.family
    fam_ok: bool | None = fam_t == fam_m or frozenset({fam_t, fam_m}) in FAMILY_EQUIV
    eligible = timing_scored and speed == "fast" and tt.duration_ms >= th["family_min_duration_ms"]
    if not eligible:
        fam_ok = None if not fam_ok else True  # informative only
    rmse = progress_rmse(tt, it.start_ms, it.duration_ms, it.easing.cubic_bezier)
    return Row(
        segment=tt.segment_id,
        element=tt.element_id,
        ir_element=iel,
        prop=tt.property,
        truth=f"{_fmt_value(tt.from_)}→{_fmt_value(tt.to)}",
        measured=f"{_fmt_value(it.from_)}→{_fmt_value(it.to)}",
        value_err=value_err,
        value_ok=value_ok,
        start_err_ms=start_err,
        start_ok=start_ok,
        dur_err_ms=dur_err,
        dur_ok=dur_ok,
        family=f"{fam_t}/{fam_m}",
        family_ok=fam_ok,
        rmse=rmse,
        rmse_ok=(rmse <= th["progress_rmse"]) if timing_scored else None,
        band=it.confidence.band,
        timing_conf=it.confidence.timing,
        eligible_family=eligible,
    )


def _relationships(truth: Truth, spec: MotionSpec, eid: dict[str, str], rep: Report) -> None:
    if not truth.relationships:
        return
    # truth transition id -> IR transition id
    tmap: dict[str, str] = {}
    for tt in truth.transitions:
        iel = eid.get(tt.element_id)
        for x in spec.transitions:
            if (x.element_id, x.property, x.segment_id) == (iel, tt.property, tt.segment_id):
                tmap[tt.id] = x.id
    for rel in truth.relationships:
        want_ids = {tmap.get(i) for i in rel.transition_ids}
        cands = [
            r for r in spec.relationships
            if r.kind == rel.kind and r.segment_id == rel.segment_id
            and len(want_ids & set(r.transition_ids)) >= 2
        ]  # fmt: skip
        what = f"relationship {rel.segment_id} {rel.kind}"
        if not cands:
            rep.checks.append((what, False, f"not found for {rel.transition_ids}"))
            continue
        if rel.kind == "stagger":
            got = cands[0].interval_ms or 0
            ok = abs(got - (rel.interval_ms or 0)) <= THRESHOLDS["stagger_interval_ms"]
            rep.checks.append((what, ok, f"interval want {rel.interval_ms}, got {got}"))
        else:
            rep.checks.append((what, True, ""))


# --------------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------------


def _mark(ok: bool | None) -> str:
    return "-" if ok is None else ("ok" if ok else "FAIL")


def format_report(rep: Report) -> str:
    lines = [f"== {rep.name}: {'PASS' if rep.ok else 'FAIL'}"]
    if rep.rows:
        hdr = (f"{'seg':6} {'element':9} {'ir':4} {'property':16} {'truth':22} {'measured':22} "
               f"{'value':10} {'start':>8} {'dur':>8} {'family':24} {'rmse':>6}")  # fmt: skip
        lines += [hdr, "-" * len(hdr)]
        for r in rep.rows:
            se = "" if r.start_err_ms is None else f"{r.start_err_ms:+.0f}"
            de = "" if r.dur_err_ms is None else f"{r.dur_err_ms:+.0f}"
            rm = "" if r.rmse is None else f"{r.rmse:.3f}"
            lines.append(
                f"{r.segment:6} {r.element:9} {r.ir_element or '-':4} {r.prop:16} {r.truth:22} "
                f"{r.measured:22} {r.value_err + ' ' + _mark(r.value_ok):10} "
                f"{se + ' ' + _mark(r.start_ok):>8} {de + ' ' + _mark(r.dur_ok):>8} "
                f"{r.family + ' ' + _mark(r.family_ok):24} {rm + ' ' + _mark(r.rmse_ok):>6}"
                + (f"  {r.note}" if r.note else "")
            )
    for what, ok, detail in rep.checks:
        lines.append(f"  [{_mark(ok):4}] {what}: {detail}")
    for x in rep.extra:
        lines.append(f"  [info] extra IR transition {x}")
    return "\n".join(lines)


@dataclass
class Summary:
    """Suite-level §11.4 view over many reports."""

    scenarios_ok: int
    scenarios: int
    family_correct: int
    family_eligible: int
    #: band → (rows within targets, rows)
    calibration: dict[str, tuple[int, int]]
    failures: list[str]
    #: PLAN-continuous scenarios evaluated (``suite == "continuous"``): (pass, total)
    continuous: tuple[int, int] = (0, 0)
    #: scenarios not evaluated yet (continuous suite before P2)
    pending: list[str] = field(default_factory=list)
    #: continuous values: band → (values within target, values)
    continuous_calibration: dict[str, tuple[int, int]] = field(default_factory=dict)

    @property
    def family_rate(self) -> float:
        return self.family_correct / self.family_eligible if self.family_eligible else 1.0

    @property
    def ok(self) -> bool:
        """Pending scenarios neither pass nor fail the suite."""
        return self.scenarios_ok == self.scenarios and self.family_rate >= FAMILY_MIN_RATE


#: §11.4: easing family correct on ≥ 80 % of eligible transitions.
FAMILY_MIN_RATE = 0.80


def summarize(reports: list[Report], pending: list[str] | None = None) -> Summary:
    rows = [r for rep in reports for r in rep.rows if r.ir_element is not None]
    elig = [r for r in rows if r.eligible_family]
    cal: dict[str, tuple[int, int]] = {}
    for band in ("high", "medium", "low"):
        rs = [r for r in rows if r.band == band]
        cal[band] = (sum(r.within_targets for r in rs), len(rs))
    failures: list[str] = []
    for rep in reports:
        for r in rep.rows:
            if not r.ok:
                bad = [
                    n
                    for n, v in (
                        ("value", r.value_ok),
                        ("start", r.start_ok),
                        ("duration", r.dur_ok),
                        ("rmse", r.rmse_ok),
                    )
                    if v is False
                ]
                what = ", ".join(bad) or r.note or "missing"
                failures.append(f"{rep.name}: {r.segment} {r.element} {r.prop} ({what})")
        for what, ok, detail in rep.checks:
            if ok is False:
                failures.append(f"{rep.name}: {what} ({detail})")
    return Summary(
        scenarios_ok=sum(r.ok for r in reports),
        scenarios=len(reports),
        family_correct=sum(bool(r.family_ok) for r in elig),
        family_eligible=len(elig),
        calibration=cal,
        failures=failures,
        continuous=(
            sum(r.ok for r in reports if r.suite == "continuous"),
            sum(r.suite == "continuous" for r in reports),
        ),
        pending=list(pending or []),
        continuous_calibration={
            band: (
                sum(ok for rep in reports for _, b, ok in rep.cvalues if b == band),
                sum(b == band for rep in reports for _, b, _ in rep.cvalues),
            )
            for band in ("high", "medium", "low")
        },
    )


def format_summary(s: Summary) -> str:
    lines = [f"scenarios: {s.scenarios_ok}/{s.scenarios} pass"]
    c_ok, c_n = s.continuous
    if c_n:
        lines.append(f"continuous suite (PLAN-continuous §8.4): {c_ok}/{c_n} pass")
    if s.pending:
        lines.append(
            f"continuous suite: {len(s.pending)} pending until P2 (not run, not judged): "
            + ", ".join(s.pending)
        )
    lines += [
        f"easing family (eligible, §11.4 >= {FAMILY_MIN_RATE:.0%}): "
        f"{s.family_correct}/{s.family_eligible} = {s.family_rate:.0%} "
        f"{'ok' if s.family_rate >= FAMILY_MIN_RATE else 'FAIL'}",
        "confidence calibration (rows within all targets / rows, per band):",
    ]
    for band, (ok, n) in s.calibration.items():
        rate = f"{ok / n:.0%}" if n else "-"
        lines.append(f"  {band:6} {ok}/{n} {rate}")
    if any(n for _, n in s.continuous_calibration.values()):
        lines.append("continuous calibration (values within §8.4 target / values, per band):")
        for band, (ok, n) in s.continuous_calibration.items():
            rate = f"{ok / n:.0%}" if n else "-"
            lines.append(f"  {band:6} {ok}/{n} {rate}")
    if s.failures:
        lines.append("failures:")
        lines += [f"  - {f}" for f in s.failures]
    lines.append(f"SUITE: {'PASS' if s.ok else 'FAIL'}")
    return "\n".join(lines)


def load_ir(data: dict) -> tuple[MotionSpec | None, str | None]:
    """Accept a ResultEnvelope, a bare MotionSpec, or an error body ``{"error": {"code"}}``."""
    if "error" in data and isinstance(data["error"], dict):
        return None, data["error"].get("code")
    if "spec" in data:
        data = data["spec"]
    return MotionSpec.model_validate(data), None


# --------------------------------------------------------------------------------------------
# Oracle: truth -> MotionSpec
# --------------------------------------------------------------------------------------------


_HI = 0.95
#: Appearance confidences of the oracle (font size is capped below "high" by the IR).
_COLOR_CONF, _FONT_CONF, _RADIUS_CONF = 0.85, 0.6, 0.7


def _ir_easing(e: TruthEasing) -> Easing:
    return Easing(
        keyword=e.keyword,  # type: ignore[arg-type]
        cubic_bezier=e.cubic_bezier,
        nearest_named=e.name,
        family=e.family,
        rmse=0.0,
        flags=["overshoot"] if e.family == "overshoot" else [],
    )


def _mn(value: float, conf: float = _HI) -> MeasuredNumber:
    return MeasuredNumber(value=value, confidence=Confidence.of(conf))


def _oracle_static(truth: Truth, element_id: str) -> ElementStatic:
    ap = truth.element(element_id).appearance
    if ap is None:
        return ElementStatic()
    text_like = truth.element(element_id).text_like

    def color(h: str | None) -> MeasuredColor | None:
        return MeasuredColor(value=h, confidence=Confidence.of(_COLOR_CONF)) if h else None

    return ElementStatic(
        border_radius_px=(
            _mn(ap.border_radius_px, _RADIUS_CONF) if ap.border_radius_px is not None else None
        ),
        background_color=color(ap.background_color),
        text_color=color(ap.text_color) if text_like else None,
        font_size_px=(
            _mn(ap.font_size_px, _FONT_CONF) if text_like and ap.font_size_px is not None else None
        ),
    )


def _oracle_scene(truth: Truth) -> Scene | None:
    if truth.scene is None:
        return None
    return Scene(
        viewport_css=Size(w=truth.scene.viewport_css.w, h=truth.scene.viewport_css.h),
        page_background=MeasuredColor(
            value=truth.scene.page_background, confidence=Confidence.of(_COLOR_CONF)
        ),
    )


def _oracle_source(truth: Truth) -> Source:
    v = truth.video
    probe = v.probe
    return Source(
        filename=v.file,
        container=v.container,
        codec=v.codec,
        width=v.width,
        height=v.height,
        duration_ms=probe.duration_ms if probe else v.duration_ms,
        fps_nominal=v.fps,
        fps_effective=probe.avg_fps if probe else v.fps,
        is_vfr=v.vfr,
        pixel_ratio=v.pixel_ratio,
        pixel_ratio_source="user",
        timing_resolution_ms=int(round(1000 / v.fps)),
    )


def _oracle_cursor(truth: Truth, ids: dict[str, str]) -> Cursor:
    events = [
        CursorEvent(t_ms=e.t_ms, kind=e.kind, element_id=ids.get(e.element_id or ""))  # type: ignore[arg-type]
        for e in truth.cursor.events
        if e.kind in ("enter", "leave", "stationary", "move_start")
    ]
    return Cursor(visible=truth.cursor.visible, confidence=1.0, events=events)


def oracle_spec(truth: Truth) -> MotionSpec:
    """A perfect pipeline's IR for ``truth`` (valid against the frozen contract)."""
    assert truth.interaction is not None, "negative scenarios have no IR"
    if truth.continuous is not None:
        return _oracle_continuous(truth)
    ids = {e.id: f"e{i}" for i, e in enumerate(truth.elements, start=1)}
    tids = {t.id: t.id for t in truth.transitions}
    hi = Confidence.of(_HI)

    elements = [
        MotionElement(
            id=ids[e.id],
            label=e.label,
            label_source="heuristic",
            role=e.role,
            parent_id=ids.get(e.parent_id) if e.parent_id else None,
            kind=e.kind,
            bbox_initial=e.bbox_initial,
            bbox_active=e.bbox_active,
            text_like=e.text_like,
            static=_oracle_static(truth, e.id),
        )  # fmt: skip
        for e in truth.elements
    ]
    segments = [
        Segment(
            id=s.id,
            kind=s.kind,
            start_ms=s.onset_ms,
            end_ms=s.settle_ms,
            onset_ms=s.onset_ms,
            settle_ms=s.settle_ms,
            trigger_event_ms=s.trigger_event_ms,
        )  # fmt: skip
        for s in truth.segments
    ]
    transitions = []
    for t in truth.transitions:
        bez = CubicBezier(*t.easing.cubic_bezier)
        ts = np.linspace(t.start_ms, t.start_ms + t.duration_ms, 13)
        samples = [(int(round(x)), float(bez((x - t.start_ms) / t.duration_ms))) for x in ts]
        transitions.append(
            Transition(
                id=tids[t.id],
                segment_id=t.segment_id,
                element_id=ids[t.element_id],
                property=t.property,
                **{"from": t.from_},
                to=t.to,
                delta=t.delta,
                start_ms=t.start_ms,
                delay_ms=t.delay_ms,
                duration_ms=t.duration_ms,
                easing=_ir_easing(t.easing),
                transform_origin=t.transform_origin,
                confidence=TransitionConfidence(
                    overall=0.95, value=0.95, timing=0.95, easing=0.95, band="high"
                ),
                notes=list(t.notes),
                samples=samples,
            )  # fmt: skip
        )
    rels = [
        Relationship(
            kind=r.kind,
            segment_id=r.segment_id,
            transition_ids=[tids[i] for i in r.transition_ids],
            offset_ms=r.offset_ms,
            interval_ms=r.interval_ms,
        )  # fmt: skip
        for r in truth.relationships
    ]
    seg = {s.id: s for s in truth.segments}
    fwd = seg.get("fwd") or seg["rt_in"]
    rev = seg.get("rev") or seg.get("rt_out")
    inter = truth.interaction
    warnings = []
    if inter.direction == "forward":
        warnings.append(
            SpecWarning(code="reverse_not_recorded", severity="info", message="No reverse.")
        )
    for code in truth.expected_warnings:
        warnings.append(SpecWarning(code=code, severity="info", message="synthetic"))
    return MotionSpec(
        job_id=hashlib.md5(truth.name.encode()).hexdigest(),
        meta=Meta(generated_at=datetime.now(UTC), pipeline_version="synth-oracle"),
        source=_oracle_source(truth),
        scene=_oracle_scene(truth),
        interaction=Interaction(
            type=inter.type,
            type_source="heuristic",
            type_confidence=hi,
            pattern=inter.pattern,
            target_element_id=ids.get(inter.target_element_id or ""),
            target_label=truth.element(inter.target_element_id).label
            if inter.target_element_id
            else "",
            trigger=Trigger(
                kind=inter.trigger,
                reverse_kind=inter.reverse_trigger,
                description="synthetic",
                confidence=hi,
            ),
            direction=inter.direction,
            total_duration_ms=TotalDuration(
                forward=fwd.settle_ms - fwd.onset_ms,
                reverse=(rev.settle_ms - rev.onset_ms) if rev else None,
            ),
        ),  # fmt: skip
        elements=elements,
        segments=segments,
        transitions=transitions,
        relationships=rels,
        cursor=_oracle_cursor(truth, ids),
        structure=[],
        interpretation=Interpretation(provider="none", status="disabled"),
        warnings=warnings,
    )


def _oracle_phase(i: int, tp: TruthPhase) -> Phase:
    hi = Confidence.of(_HI)
    fit = None
    if tp.kind == "autoplay":
        fit = ConstantFit(velocity_px_s=_mn(tp.v_start_px_s))
    elif tp.kind in ("decelerate", "resume"):
        assert tp.ramp_ms is not None and tp.easing is not None
        fit = RampFit(from_px_s=tp.v_start_px_s, to_px_s=tp.v_end_px_s,
                      duration_ms=_mn(tp.ramp_ms), easing=_ir_easing(tp.easing))  # fmt: skip
    elif tp.kind == "inertia":
        assert tp.tau_ms is not None and tp.v0_px_s is not None
        # the truth cuts a decay to rest at |v| = 10 px/s (its v_end): the stop speed
        stop = abs(tp.v_end_px_s) if not tp.v_inf_px_s else None
        fit = ExponentialFit(tau_ms=_mn(tp.tau_ms), v0_px_s=_mn(tp.v0_px_s),
                             v_inf_px_s=tp.v_inf_px_s or 0.0,
                             stop_px_s=round(stop, 1) if stop else None)  # fmt: skip
    elif tp.kind == "snap":
        assert tp.snap_distance_px is not None and tp.easing is not None
        fit = TweenFit(
            distance_px=_mn(tp.snap_distance_px),
            duration_ms=_mn(tp.end_ms - tp.start_ms),
            easing=_ir_easing(tp.easing),
        )
    return Phase(
        id=f"p{i}",
        kind=tp.kind,
        start_ms=tp.start_ms,
        end_ms=tp.end_ms,
        v_start_px_s=tp.v_start_px_s,
        v_end_px_s=tp.v_end_px_s,
        v_peak_px_s=tp.v_peak_px_s,
        displacement_px=tp.displacement_px,
        fit=fit,
        interrupted=tp.interrupted,
        confidence=hi,
    )


def _oracle_behavior(tc: TruthContinuous) -> Behavior:
    hi = Confidence.of(_HI)
    ph = tc.phases

    def first(kind: str) -> TruthPhase | None:
        return next((p for p in ph if p.kind == kind), None)

    decel, resume, snap = first("decelerate"), first("resume"), first("snap")
    inertias = [p for p in ph if p.kind == "inertia"]
    snaps = [p for p in ph if p.kind == "snap"]
    pause = None
    if tc.pause_on is not None:
        pause = PauseBehavior(
            on=tc.pause_on_accept[0] if tc.pause_on_accept else tc.pause_on,
            on_confidence=hi,
            decel_ms=_mn(tc.pause_decel_ms) if tc.pause_decel_ms is not None else None,
            easing=_ir_easing(decel.easing) if decel is not None and decel.easing else None,
            stops_completely=True,
        )
    drag = None
    if tc.drag_count:
        assert tc.drag_peak_speed_px_s is not None
        drag = DragBehavior(
            count=tc.drag_count,
            follows_pointer=tc.drag_follows_pointer,
            pointer_ratio=_mn(tc.drag_pointer_ratio) if tc.drag_pointer_ratio else None,
            peak_speed_px_s=_mn(tc.drag_peak_speed_px_s),
        )
    inertia = None
    if inertias:
        v0s = [abs(p.v0_px_s or 0.0) for p in inertias]
        assert tc.inertia_tau_ms is not None
        inertia = InertiaBehavior(
            model="exponential", tau_ms=_mn(tc.inertia_tau_ms), duration_ms=None, easing=None,
            release_speed_min_px_s=min(v0s), release_speed_max_px_s=max(v0s),
            instances=len(inertias),
        )  # fmt: skip
    snap_b = None
    if tc.snap_kind == "grid" and snap is not None:
        durs = sorted(p.end_ms - p.start_ms for p in snaps)
        snap_b = SnapBehavior(
            kind="grid", step_px=_mn(tc.pitch_px), duration_ms=_mn(durs[len(durs) // 2]),
            easing=_ir_easing(snap.easing) if snap.easing else None, overshoot=False,
        )  # fmt: skip
    resume_b = None
    if resume is not None:
        assert tc.resume_delay_after_rest_ms is not None and tc.resume_ramp_ms is not None
        rel = tc.resume_delay_after_release_ms
        resume_b = ResumeBehavior(
            delay_after_rest_ms=_mn(tc.resume_delay_after_rest_ms),
            delay_after_release_ms=_mn(rel) if rel is not None else None,
            delay_after_leave_ms=(
                _mn(tc.resume_delay_after_leave_ms)
                if tc.resume_delay_after_leave_ms is not None
                else None
            ),
            ramp_ms=_mn(tc.resume_ramp_ms),
            easing=_ir_easing(resume.easing) if resume.easing else None,
            to_speed_px_s=abs(tc.autoplay_velocity_px_s or 0.0),
            direction_preserved=bool(tc.resume_direction_preserved),
        )
    return Behavior(pause=pause, drag=drag, inertia=inertia, snap=snap_b, resume=resume_b)


def _oracle_card_scale(tc: TruthContinuous) -> CardScale | None:
    """The truth lens as IR: reference at 70 % of the half viewport (a whole card's centre)."""
    z = tc.zoom
    if z is None:
        return None
    half = (tc.region.w if tc.axis == "x" else tc.region.h) / 2.0
    ref = round(0.7 * half, 1)
    at_ref = round(1.0 + (z.edge_scale - 1.0) * (ref / half) ** 2, 4)
    return CardScale(
        reference_distance_px=ref,
        scale_at_reference=at_ref,
        mean_scale=round(1.0 + (at_ref - 1.0) * (half / ref) ** 2 / 3.0, 4),
        confidence=Confidence.of(0.6),
    )


def _oracle_continuous(truth: Truth) -> MotionSpec:
    tc = truth.continuous
    inter = truth.interaction
    assert tc is not None and inter is not None
    hi = Confidence.of(_HI)
    ids = {e.id: f"e{i}" for i, e in enumerate(truth.elements, start=1)}
    scroller = ids[tc.element_id]
    elements = [
        MotionElement(
            id=ids[e.id],
            label=e.label,
            label_source="heuristic",
            role=e.role,
            parent_id=ids.get(e.parent_id) if e.parent_id else None,
            kind=e.kind,
            bbox_initial=e.bbox_initial,
            bbox_active=e.bbox_active,
            text_like=e.text_like,
            static=_oracle_static(truth, e.id),
        )  # fmt: skip
        for e in truth.elements
    ]
    # the card (P2b: required, §14): its size from the truth, placed at the scroller's start
    r = tc.region
    elements.append(
        MotionElement(
            id=f"e{len(elements) + 1}",
            label="Card",
            label_source="heuristic",
            role="card",
            parent_id=scroller,
            kind="transform",
            bbox_initial=Box(x=r.x, y=r.y, w=tc.card.w, h=tc.card.h),
            bbox_active=Box(x=r.x, y=r.y, w=tc.card.w, h=tc.card.h),
            static=ElementStatic(
                background_color=MeasuredColor(
                    value=tc.card.background_color, confidence=Confidence.of(_COLOR_CONF)
                ),
            ),  # fmt: skip
        )
    )
    autoplay = None
    v = tc.autoplay_velocity_px_s
    if v is not None:
        speed = abs(v)
        obs = tc.loop.observable
        autoplay = Autoplay(
            direction=tc.autoplay_direction,  # type: ignore[arg-type]
            speed_px_s=_mn(speed),
            velocity_px_s=v,
            loop=LoopInfo(
                observed=obs,
                period_px=_mn(tc.loop.period_px, 0.9) if obs else None,
                duration_ms=_mn(tc.loop.period_px / speed * 1000.0, 0.9) if obs else None,
            ),
        )
    rows = tc.profile
    step = max(1, -(-len(rows) // 900))  # ceil: keep <= CONTINUOUS_SAMPLE_MAX samples
    samples = [(t, vel, pos, 1.0) for t, pos, vel in rows[::step]]
    start, end = tc.span_ms
    warnings = []
    if v is not None and not tc.loop.observable:
        warnings.append(SpecWarning(code="loop_period_not_observed", severity="info",
                                    message="synthetic"))  # fmt: skip
    for code in truth.expected_warnings:
        warnings.append(SpecWarning(code=code, severity="info", message="synthetic"))
    return MotionSpec(
        job_id=hashlib.md5(truth.name.encode()).hexdigest(),
        meta=Meta(generated_at=datetime.now(UTC), pipeline_version="synth-oracle"),
        mode="continuous",
        source=_oracle_source(truth),
        scene=_oracle_scene(truth),
        interaction=Interaction(
            type=inter.type,
            type_source="heuristic",
            type_confidence=hi,
            pattern=inter.pattern,
            target_element_id=scroller,
            target_label=truth.element(tc.element_id).label,
            trigger=Trigger(
                kind=inter.trigger,
                reverse_kind=inter.reverse_trigger,
                description="synthetic",
                confidence=hi,
            ),  # fmt: skip
            direction="continuous",
            total_duration_ms=TotalDuration(forward=end - start, reverse=None),
        ),
        elements=elements,
        segments=[],
        transitions=[],
        relationships=[],
        continuous=ContinuousMotion(
            element_id=scroller,
            axis=tc.axis,
            region=Box(x=tc.region.x, y=tc.region.y, w=tc.region.w, h=tc.region.h),
            region_confidence=Confidence.of(0.7),
            autoplay=autoplay,
            pitch_px=_mn(tc.pitch_px, 0.8),
            gap_px=_mn(tc.gap_px, 0.5),
            card_scale=_oracle_card_scale(tc),
            phases=[_oracle_phase(i, p) for i, p in enumerate(tc.phases, start=1)],
            behavior=_oracle_behavior(tc),
            span_ms=Span(start_ms=start, end_ms=end),
            samples=samples,
        ),
        cursor=_oracle_cursor(truth, ids),
        structure=[],
        interpretation=Interpretation(provider="none", status="disabled"),
        warnings=warnings,
    )
