"""Phase 4 integration logic: IR assembly + merge policy (§7.5) and run.py helpers."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from app.interpret.base import (
    ElementSummary,
    InterpretationInput,
    InterpretationResult,
    InterpretedElement,
    InterpretedInteraction,
    VideoMeta,
)
from app.interpret.fallback import build_fallback
from app.models.ir import Box, PxValue, RatioValue
from app.models.measure import (
    ElementCandidate,
    FittedTransition,
    HeuristicInteraction,
    ProbeInfo,
    PropertySeries,
    Rect,
    Scale,
)
from app.pipeline import assemble as asm
from app.pipeline.confidence import make_transition_confidence
from app.pipeline.easing import CANDIDATES_BY_NAME, eval_curve, make_easing
from app.pipeline.params import DEFAULT_PARAMS
from app.pipeline.run import (
    EXTRAPOLATED_NOTE,
    filter_series,
    shared_timing_endpoint,
    tail_factor,
)
from app.pipeline.timing import fit_series


def _series(eid: str, prop: str, seg: str = "fwd", v0: float = 0.0, v1: float = -8.0,
            notes: list[str] | None = None) -> PropertySeries:  # fmt: skip
    t = np.linspace(0.8, 1.6, 49)
    p = np.clip((t - 1.0) / 0.28, 0, 1)
    cls = PxValue if prop.startswith("translate") or prop == "height" else RatioValue
    return PropertySeries(
        eid, prop, seg, t, v0 + (v1 - v0) * p, np.full(t.size, 0.99), v0, v1,  # type: ignore[arg-type]
        "px" if cls is PxValue else "ratio", cls(number=v0), cls(number=v1),
        notes=list(notes or []),
    )  # fmt: skip


_BOX = Rect(100, 100, 300, 200)


def _el(
    eid: str,
    kind: str = "transform",
    parent: str | None = None,
    box: Rect = _BOX,
    notes: list[str] | None = None,
) -> ElementCandidate:
    return ElementCandidate(eid, box, box, parent, kind, 1.0, notes=list(notes or []))  # type: ignore[arg-type]


def _ft(tid: str, s: PropertySeries, t0: float = 1.0, d: float = 0.28) -> FittedTransition:
    return FittedTransition(
        tid,
        s,
        t0,
        d,
        make_easing((0, 0, 0.58, 1), 0.01, named=CANDIDATES_BY_NAME["ease-out"]),
        make_transition_confidence(0.9, 0.9, 0.9, cap=0.95),
    )


def _measurement(elements: list[ElementCandidate], fts: list[FittedTransition],
                 htype: str = "hover", hconf: float = 0.85) -> asm.Measurement:  # fmt: skip
    info = ProbeInfo("mp4", "h264", 1280, 800, 0, 4.0, np.arange(0, 4, 1 / 60), 60.0, 60.0,
                     False, False)  # fmt: skip
    return asm.Measurement(
        probe=info, scale=Scale(1, "user", 1.0), frame_css=(1280, 800),
        direction="forward", segments=[asm.SegmentWindow("fwd", 0.98, 1.35)],
        elements=elements, transitions=fts,
        heuristic=HeuristicInteraction(htype, "pointer_enter", "none", "forward", hconf, 5),  # type: ignore[arg-type]
        target_id=elements[0].id, cursor_visible=True, cursor_confidence=0.9,
        cursor_events=[], cursor_summary=f"enters {elements[0].id} at 1000 ms",
        timing_resolution_ms=17,
    )  # fmt: skip


def _interp(m: asm.Measurement, tmp_path: Path) -> InterpretationResult:
    return build_fallback(asm.interpretation_input(m, tmp_path, {}), status="disabled")


def _assemble(m: asm.Measurement, interp: InterpretationResult, use: bool = False):
    return asm.assemble(m, interp, job_id="a" * 32, filename="x.mp4",
                        generated_at=datetime.now(UTC), pipeline_version="t",
                        use_interpreter=use)  # fmt: skip


def test_assemble_minimal_spec(tmp_path: Path) -> None:
    els = [_el("e1")]
    m = _measurement(els, [_ft("t1", _series("e1", "translateY"))])
    spec = _assemble(m, _interp(m, tmp_path))
    assert spec.interaction.type == "hover" and spec.interaction.direction == "forward"
    assert {w.code for w in spec.warnings} == {"reverse_not_recorded"}
    t = spec.transitions[0]
    assert (t.start_ms, t.duration_ms, t.delta) == (1000, 280, -8.0)
    assert spec.segments[0].onset_ms == 1000 and spec.segments[0].settle_ms == 1280
    assert spec.elements[0].label_source == "heuristic" and t.samples


def test_sub_frame_delay_is_zero(tmp_path: Path) -> None:
    els = [_el("e1"), _el("e2", parent="e1", box=Rect(110, 110, 50, 50))]
    fts = [
        _ft("t1", _series("e1", "translateY")),
        _ft("t2", _series("e2", "scale", v0=1.0, v1=1.06), t0=1.010),
        _ft("t3", _series("e2", "translateX", v0=0.0, v1=4.0), t0=1.030),
    ]
    spec = _assemble(_measurement(els, fts), _interp(_measurement(els, fts), tmp_path))
    delays = {t.id: t.delay_ms for t in spec.transitions}
    assert delays == {"t1": 0, "t2": 0, "t3": 30}  # 10 ms < one 17 ms frame


def test_merge_type_policy() -> None:
    h = HeuristicInteraction("hover", "pointer_enter", "none", "forward", 0.85, 5)
    h_weak = HeuristicInteraction("unknown", "unknown", "none", "forward", 0.3, 8)

    def res(t: str, c: float, status: str = "ok") -> InterpretationResult:
        return InterpretationResult(
            status=status, provider="claude_cli", elements={},  # type: ignore[arg-type]
            interaction=InterpretedInteraction(type_confirmation=t, type_confidence=c),  # type: ignore[arg-type]
        )  # fmt: skip

    assert asm.merge_type(h, res("click", 0.9)) == ("hover", "heuristic", 0.85, True)
    assert asm.merge_type(h_weak, res("dropdown", 0.8))[:2] == ("dropdown", "interpreter")
    assert asm.merge_type(h_weak, res("dropdown", 0.6))[:2] == ("unknown", "heuristic")
    assert asm.merge_type(h, res("hover", 0.9)) == ("hover", "heuristic", 0.9, False)
    # fallback results report type_confidence 0 -> the heuristic confidence is used
    assert asm.merge_type(h, res("hover", 0.0, "timeout")) == ("hover", "heuristic", 0.85, False)


def test_backdrop_is_never_a_css_parent(tmp_path: Path) -> None:
    els = [
        _el("e1", "backdrop", box=Rect(0, 0, 1280, 800)),
        _el("e2", "appear", parent="e1", box=Rect(400, 250, 480, 300)),
    ]
    fts = [_ft("t1", _series("e1", "opacity", v0=0, v1=0.5)),
           _ft("t2", _series("e2", "opacity", v0=0, v1=1))]  # fmt: skip
    m = _measurement(els, fts, htype="modal")
    interp = _interp(m, tmp_path)
    # an "ok" interpreter that nests the panel inside the backdrop again is ignored
    ok = interp.model_copy(update={"status": "ok", "elements": {
        "e1": interp.elements["e1"],
        "e2": interp.elements["e2"].model_copy(update={"parent_id": "e1"}),
    }})  # fmt: skip
    for r in (interp, ok):
        spec = _assemble(m, r)
        assert next(e for e in spec.elements if e.id == "e2").parent_id is None


def test_interpreter_reparent_guarded_by_motion(tmp_path: Path) -> None:
    els = [_el("e1", box=Rect(100, 100, 300, 300)), _el("e2", "photometric",
           box=Rect(120, 120, 60, 20))]  # fmt: skip
    fts = [_ft("t1", _series("e1", "translateY")),
           _ft("t2", _series("e2", "opacity", v0=0.5, v1=1.0))]  # fmt: skip
    m = _measurement(els, fts)
    base = _interp(m, tmp_path)
    ok = base.model_copy(update={"status": "ok", "elements": {
        "e1": base.elements["e1"],
        "e2": InterpretedElement(label="Title", role="title", parent_id="e1", confidence=0.9),
    }})  # fmt: skip
    # e1 moves: re-parenting e2 under it would make e2's absolute values relative -> refused
    assert asm.merge_parents(els, ok, fts)["e2"] is None
    still = [_ft("t1", _series("e1", "opacity", v0=0.5, v1=1.0)), fts[1]]
    assert asm.merge_parents(els, ok, still)["e2"] == "e1"


def test_label_source_per_element(tmp_path: Path) -> None:
    els = [_el("e1"), _el("e2", "photometric", box=Rect(120, 120, 60, 20))]
    fts = [_ft("t1", _series("e1", "translateY")),
           _ft("t2", _series("e2", "opacity", v0=0.5, v1=1.0))]  # fmt: skip
    m = _measurement(els, fts)
    base = _interp(m, tmp_path)
    ok = base.model_copy(update={"status": "ok", "elements": {
        "e1": InterpretedElement(label="Product card", role="card", confidence=0.9),
        "e2": base.elements["e2"],  # omitted by the model -> heuristic label kept
    }})  # fmt: skip
    spec = _assemble(m, ok, use=True)
    src = {e.id: e.label_source for e in spec.elements}
    assert src == {"e1": "interpreter", "e2": "heuristic"}


def test_interpretation_fallback_warning_only_when_opted_in(tmp_path: Path) -> None:
    m = _measurement([_el("e1")], [_ft("t1", _series("e1", "translateY"))])
    timeout = build_fallback(asm.interpretation_input(m, tmp_path, {}), status="timeout")
    assert "interpretation_fallback" in {w.code for w in _assemble(m, timeout, use=True).warnings}
    disabled = _interp(m, tmp_path)
    assert "interpretation_fallback" not in {w.code for w in _assemble(m, disabled).warnings}


def test_filter_series_drops_artefact_scales() -> None:
    els = [_el("e1", "appear"), _el("e2"), _el("e3", notes=["caused_by_resize"])]
    series = [
        _series("e1", "translateY"), _series("e1", "scale", v0=1.01, v1=1.0),
        _series("e1", "scaleX", v0=1.0, v1=1.0), _series("e1", "scaleY", v0=1.02, v1=1.0),
        _series("e2", "scale", v0=1.0, v1=1.05), _series("e2", "scaleX", v0=1.0, v1=1.1),
        _series("e2", "scaleY", v0=1.0, v1=1.0),
        _series("e3", "translateY", v1=120), _series("e3", "scale", v0=1.0, v1=0.99),
    ]  # fmt: skip
    kept = {(s.element_id, s.property) for s in filter_series(series, els)}
    assert kept == {("e1", "translateY"), ("e2", "scaleX"), ("e2", "scaleY"),
                    ("e3", "translateY")}  # fmt: skip


def test_tail_factor_penalizes_strong_ease_out_only() -> None:
    assert tail_factor(CANDIDATES_BY_NAME["ease-out"].bezier, DEFAULT_PARAMS) == 1.0
    assert tail_factor(CANDIDATES_BY_NAME["ease-in-out"].bezier, DEFAULT_PARAMS) == 1.0
    expo = tail_factor(CANDIDATES_BY_NAME["easeOutExpo"].bezier, DEFAULT_PARAMS)
    assert expo == pytest.approx(DEFAULT_PARAMS.confidence.tail_min_factor)


def test_shared_timing_endpoint_recovers_invisible_start() -> None:
    """Slide-in seen only while visible: the start value comes from the fitted opacity curve."""
    bez = CANDIDATES_BY_NAME["expoOut"].bezier
    t = np.arange(0.8, 1.8, 1 / 60)
    p = eval_curve(bez, np.clip((t - 1.2) / 0.2, 0, 1))
    rng = np.random.default_rng(3)
    op = PropertySeries("e1", "opacity", "fwd", t, p + rng.normal(0, 0.02, t.size),
                        np.ones(t.size), 0.0, 1.0, "ratio", RatioValue(number=0),
                        RatioValue(number=1))  # fmt: skip
    vis = p > 0.3
    ty = PropertySeries("e1", "translateY", "fwd", t[vis], -8 * (1 - p[vis]), np.ones(vis.sum()),
                        -7.0, 0.0, "px", PxValue(number=-7.0), PxValue(number=0.0),
                        notes=[EXTRAPOLATED_NOTE])  # fmt: skip
    fit = fit_series(op)
    assert fit is not None
    v0, v1 = shared_timing_endpoint(ty, op, fit)  # type: ignore[misc]
    assert v1 == 0.0 and v0 == pytest.approx(-8.0, abs=0.5)


def test_fallback_numbers_repeated_roles_and_list_items(tmp_path: Path) -> None:
    els = [
        ElementSummary(
            id=f"e{i}", bbox=Box(x=520, y=y, w=280, h=48), kind="appear", change_summary="appears"
        )  # fmt: skip
        for i, y in ((1, 385), (2, 329), (3, 217), (4, 273))
    ]
    inp = InterpretationInput(
        job_dir=tmp_path, keyframe_paths={}, elements=els, heuristic_type="dropdown",
        heuristic_trigger="click", cursor_summary="cursor not visible",
        video_meta=VideoMeta(width=1280, height=800, duration_ms=3000, pixel_ratio=1),
    )  # fmt: skip
    r = build_fallback(inp, status="disabled")
    assert {e.role for e in r.elements.values()} == {"list_item"}
    assert r.elements["e3"].label == "List item 1 (e3)"  # topmost first
    assert r.elements["e1"].label == "List item 4 (e1)"
