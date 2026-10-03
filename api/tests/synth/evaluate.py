"""Compare a pipeline result (IR) against synthetic ground truth (PLAN §11.4).

* :func:`compare` — truth + MotionSpec (or a pipeline error code) → :class:`Report` with one
  row per truth transition plus interaction / relationship / error checks.
* :func:`oracle_spec` — the truth expressed as a valid ``MotionSpec`` (perfect pipeline). Used
  to self-test the evaluator and to prove the truth format fits the frozen IR contract.

Easing curves are evaluated with the suite's independent :class:`synth.animate.CubicBezier`
(never ``app.pipeline.easing``). Element ids are matched by bounding-box IoU.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import cv2
import numpy as np

from app.models.ir import (
    Confidence,
    Cursor,
    CursorEvent,
    Easing,
    Interaction,
    Interpretation,
    Meta,
    MotionElement,
    MotionSpec,
    Relationship,
    Segment,
    Source,
    SpecWarning,
    TotalDuration,
    Transition,
    TransitionConfidence,
    Trigger,
)

from .animate import CubicBezier
from .truth import Truth, TruthTransition

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

#: Family confusions not penalized (PLAN Appendix B note).
FAMILY_EQUIV = {frozenset({"ease", "ease-in-out"})}


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
    checks: list[tuple[str, bool | None, str]] = field(default_factory=list)  # (what, ok, detail)
    extra: list[str] = field(default_factory=list)

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


# --------------------------------------------------------------------------------------------
# Compare
# --------------------------------------------------------------------------------------------


def compare(truth: Truth, spec: MotionSpec | None, error_code: str | None = None) -> Report:
    rep = Report(name=truth.name)
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
    return rep


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

    @property
    def family_rate(self) -> float:
        return self.family_correct / self.family_eligible if self.family_eligible else 1.0

    @property
    def ok(self) -> bool:
        return self.scenarios_ok == self.scenarios and self.family_rate >= FAMILY_MIN_RATE


#: §11.4: easing family correct on ≥ 80 % of eligible transitions.
FAMILY_MIN_RATE = 0.80


def summarize(reports: list[Report]) -> Summary:
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
    )


def format_summary(s: Summary) -> str:
    lines = [
        f"scenarios: {s.scenarios_ok}/{s.scenarios} pass",
        f"easing family (eligible, §11.4 >= {FAMILY_MIN_RATE:.0%}): "
        f"{s.family_correct}/{s.family_eligible} = {s.family_rate:.0%} "
        f"{'ok' if s.family_rate >= FAMILY_MIN_RATE else 'FAIL'}",
        "confidence calibration (rows within all targets / rows, per band):",
    ]
    for band, (ok, n) in s.calibration.items():
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


def oracle_spec(truth: Truth) -> MotionSpec:
    """A perfect pipeline's IR for ``truth`` (valid against the frozen contract)."""
    assert truth.interaction is not None, "negative scenarios have no IR"
    ids = {e.id: f"e{i}" for i, e in enumerate(truth.elements, start=1)}
    tids = {t.id: t.id for t in truth.transitions}
    v = truth.video
    hi = Confidence.of(0.95)

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
                easing=Easing(
                    keyword=t.easing.keyword,
                    cubic_bezier=t.easing.cubic_bezier,
                    nearest_named=t.easing.name,
                    family=t.easing.family,
                    rmse=0.0,
                    flags=["overshoot"] if t.easing.family == "overshoot" else [],
                ),
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
    events = [
        CursorEvent(t_ms=e.t_ms, kind=e.kind, element_id=ids.get(e.element_id or ""))  # type: ignore[arg-type]
        for e in truth.cursor.events
        if e.kind in ("enter", "leave", "stationary", "move_start")
    ]
    warnings = []
    if inter.direction == "forward":
        warnings.append(
            SpecWarning(code="reverse_not_recorded", severity="info", message="No reverse.")
        )
    probe = v.probe
    return MotionSpec(
        job_id=hashlib.md5(truth.name.encode()).hexdigest(),
        meta=Meta(generated_at=datetime.now(UTC), pipeline_version="synth-oracle"),
        source=Source(
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
        ),  # fmt: skip
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
        cursor=Cursor(visible=truth.cursor.visible, confidence=1.0, events=events),
        structure=[],
        interpretation=Interpretation(provider="none", status="disabled"),
        warnings=warnings,
    )
