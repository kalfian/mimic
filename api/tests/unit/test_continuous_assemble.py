"""Continuous measurement → IR 0.2 (PLAN-continuous §4.7, §5): ``assemble_continuous`` emits a
valid ``MotionSpec`` for a fixture-shaped measurement (``docs/contract/
sample-continuous-result.json`` geometry / §0.1 velocity profile, no cursor, VFR)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from app.interpret.base import InterpretationResult, InterpretedElement, InterpretedInteraction
from app.interpret.fallback import FallbackInterpreter
from app.models.ir import CONTINUOUS_SAMPLE_MAX, MotionSpec
from app.models.measure import (
    AmbientRegion,
    ContinuousMeasurement,
    HeuristicInteraction,
    PhaseCandidate,
    ProbeInfo,
    Rect,
    RegimeReport,
    Scale,
    ScrollerAnalysis,
)
from app.pipeline.assemble import Measurement
from app.pipeline.continuous import assemble as cas
from app.pipeline.continuous.phases import segment
from tests.unit.test_continuous_kinematics import make_series, section01_series

JOB = "7c41e2a95b0d4f3e8a6c1b2d3e4f5a6b"
NOW = datetime(2026, 10, 3, 10, 0, tzinfo=UTC)


def _probe(dur: float = 9.3, fps: float = 55.7, vfr: bool = True) -> ProbeInfo:
    n = int(dur * fps)
    return ProbeInfo(
        container="mov", codec="h264", width=1280, height=800, rotation=0, duration_s=dur,
        frame_ts=np.arange(n) / fps, fps_nominal=60.0, fps_effective=fps, is_vfr=vfr,
        has_audio=False,
    )  # fmt: skip


def _analysis(
    series, *, pitch: float | None = 216.0, loop: float | None = None
) -> ScrollerAnalysis:
    seg = segment(series, is_vfr=True, pitch_px=pitch, pitch_confidence=0.7)
    return ScrollerAnalysis(
        region=AmbientRegion(Rect(252, 292, 776, 216), cells=600, lead_frac=0.95),
        is_scroller=True,
        axis=series.axis,
        coherence=0.97,
        series=series,
        autoplay_velocity=seg.autoplay_velocity,
        autoplay_confidence=seg.autoplay_confidence,
        loop_period_px=loop,
        loop_confidence=0.85 if loop else 0.0,
        pitch_px=pitch,
        pitch_confidence=0.7 if pitch else 0.0,
        gap_px=16.0 if pitch else None,
        gap_confidence=0.5 if pitch else 0.0,
        phases=seg.phases,
        behavior=seg.behavior,
        warnings=list(seg.warnings),
    )


def _measurement(sa: ScrollerAnalysis, probe: ProbeInfo | None = None) -> Measurement:
    probe = probe or _probe()
    regime = RegimeReport(
        ambient=[sa.region], cell_css=16.0, grid_shape=(50, 80), lead_window_s=(0.0, 0.6)
    )
    return Measurement(
        probe=probe,
        scale=Scale(pixel_ratio=1, pixel_ratio_source="auto", analysis_scale=1.0),
        frame_css=(1280.0, 800.0),
        direction="continuous",
        segments=[],
        elements=[],
        transitions=[],
        heuristic=cas.continuous_heuristic(sa),
        target_id="e1",
        cursor_visible=False,
        cursor_confidence=0.0,
        cursor_events=[],
        cursor_summary="no cursor visible",
        timing_resolution_ms=18,
        continuous=ContinuousMeasurement(regime=regime, scroller=sa),
    )


def _assemble(m: Measurement, interp: InterpretationResult | None = None) -> MotionSpec:
    if interp is None:
        inp = cas.interpretation_input_continuous(m, Path("/tmp"), {})
        interp = FallbackInterpreter().interpret(inp)
    return cas.assemble_continuous(
        m, interp, job_id=JOB, filename="carousel-drag.mov", generated_at=NOW,
        pipeline_version="0.1.0", use_interpreter=False,
    )  # fmt: skip


def _kinds(spec: MotionSpec, drop: tuple[str, ...] = ("paused",)) -> list[str]:
    assert spec.continuous is not None
    out: list[str] = []
    for p in spec.continuous.phases:
        if p.kind not in drop and (not out or out[-1] != p.kind):
            out.append(p.kind)
    return out


@pytest.fixture(scope="module")
def fixture_shaped() -> tuple[Measurement, MotionSpec]:
    m = _measurement(_analysis(section01_series(0)))
    return m, _assemble(m)


def test_fixture_shaped_measurement_emits_valid_ir(
    fixture_shaped, sample_continuous_result
) -> None:
    _, spec = fixture_shaped
    # round trip through JSON re-validates every IR rule (phase ids, behaviour ⇔ phases, mode)
    dumped = spec.model_dump(mode="json", by_alias=True)
    again = MotionSpec.model_validate(json.loads(json.dumps(dumped)))
    assert again.model_dump(mode="json", by_alias=True) == dumped

    fixture = sample_continuous_result["spec"]
    assert spec.mode == fixture["mode"] == "continuous"
    assert spec.schema_version == "0.2"
    assert spec.segments == [] and spec.transitions == [] and spec.relationships == []
    # same top-level shape as the hand-authored fixture (``scene`` comes from Track A)
    assert set(dumped) <= set(fixture)
    assert set(dumped["continuous"]) == set(fixture["continuous"])
    fixture_kinds = [p["kind"] for p in fixture["continuous"]["phases"] if p["kind"] != "paused"]
    assert _kinds(spec) == fixture_kinds

    it = spec.interaction
    assert (it.type, it.pattern, it.direction) == ("drag", "carousel", "continuous")
    assert it.trigger.kind == "drag" and it.trigger.reverse_kind == "none"
    assert it.type_source == "heuristic" and it.target_element_id == "e1"
    c = spec.continuous
    assert c is not None and c.element_id == "e1" and c.axis == "x"
    assert spec.elements[0].kind == "scroller" and spec.elements[0].role == "scroller"
    assert it.total_duration_ms.forward == c.span_ms.end_ms - c.span_ms.start_ms
    assert c.autoplay is not None and c.autoplay.direction == "right"
    assert c.autoplay.speed_px_s.value == pytest.approx(39.0, rel=0.03)
    assert not c.autoplay.loop.observed
    assert c.pitch_px is not None and c.pitch_px.value == 216.0
    assert c.behavior.drag is not None and c.behavior.drag.count == 4
    assert c.behavior.inertia is not None and c.behavior.inertia.instances == 3
    assert c.behavior.snap is not None and c.behavior.snap.kind == "abrupt_ambiguous"
    codes = {w.code for w in spec.warnings}
    assert "cursor_not_visible" in codes


def test_samples_are_decimated_and_increasing(fixture_shaped) -> None:
    _, spec = fixture_shaped
    assert spec.continuous is not None
    samples = spec.continuous.samples
    assert 0 < len(samples) <= CONTINUOUS_SAMPLE_MAX
    ts = [s[0] for s in samples]
    assert all(b > a for a, b in zip(ts, ts[1:], strict=False))
    assert all(0.0 <= s[3] <= 1.0 for s in samples)


def test_long_series_is_decimated_to_cap() -> None:
    s = make_series(lambda t: np.full_like(t, 40.0), fps=120.0, dur=9.0, sigma=0.05)
    spec = _assemble(_measurement(_analysis(s, pitch=None), _probe(9.0, 120.0, False)))
    assert spec.continuous is not None
    assert len(spec.continuous.samples) == CONTINUOUS_SAMPLE_MAX


def test_marquee_with_observed_loop() -> None:
    s = make_series(lambda t: np.full_like(t, -240.0), fps=60.0, dur=8.0, sigma=0.05)
    sa = _analysis(s, loop=1728.0)
    spec = _assemble(_measurement(sa, _probe(8.0, 60.0, False)))
    it = spec.interaction
    assert (it.type, it.pattern, it.trigger.kind) == ("continuous", "marquee", "autoplay")
    c = spec.continuous
    assert c is not None and c.autoplay is not None
    assert c.autoplay.direction == "left" and c.autoplay.velocity_px_s < 0
    loop = c.autoplay.loop
    assert loop.observed and loop.period_px is not None and loop.duration_ms is not None
    assert loop.duration_ms.value == pytest.approx(1728.0 / 240.0 * 1000.0, rel=0.03)
    assert [p.kind for p in c.phases] == ["autoplay"]
    assert c.behavior.drag is None and c.behavior.pause is None


def test_interpretation_input_is_words_only(fixture_shaped) -> None:
    m, _ = fixture_shaped
    inp = cas.interpretation_input_continuous(m, Path("/tmp"), {})
    assert inp.heuristic_type == "drag" and inp.heuristic_trigger == "drag"
    el = inp.elements[0]
    assert el.id == "e1" and el.kind == "scroller"
    assert el.change_summary.startswith("scroller; content moves horizontally (right)")
    assert "about 39 px/s" in el.change_summary and "4 drags" in el.change_summary
    assert "τ" not in el.change_summary and "tau" not in el.change_summary


def test_interpreter_labels_scroller_but_not_numbers_or_type(fixture_shaped) -> None:
    m, _ = fixture_shaped
    interp = InterpretationResult(
        status="ok",
        provider="claude_cli",
        model="test",
        duration_ms=10,
        elements={
            "e1": InterpretedElement(label="Product carousel", role="list_item", confidence=0.9)
        },
        interaction=InterpretedInteraction(
            type_confirmation="hover",
            type_confidence=0.95,
            target_element_id="e1",
            trigger_description="User drags the carousel",
        ),  # fmt: skip
        structure=["track of product cards"],
    )
    spec = _assemble(m, interp)
    assert spec.elements[0].label == "Product carousel"
    assert spec.elements[0].role == "scroller"  # role is fixed by the measurement
    assert spec.interaction.type == "drag" and spec.interaction.type_source == "heuristic"
    assert spec.interaction.trigger.description == "User drags the carousel"
    assert spec.structure == ["track of product cards"]


def test_ir_phases_keep_contiguous_ms_boundaries() -> None:
    def pc(kind: str, a: float, b: float) -> PhaseCandidate:
        return PhaseCandidate(kind=kind, start_s=a, end_s=b, v_start=0, v_end=0, v_peak=0,
                              displacement=0, confidence=0.5)  # fmt: skip

    out = cas.ir_phases(
        [pc("paused", 0.0, 1.0), pc("paused", 1.0, 1.0003), pc("paused", 1.0003, 2.0)],
        (0, 2000),
    )
    assert [p.id for p in out] == ["p1", "p2", "p3"]
    assert [(p.start_ms, p.end_ms) for p in out] == [(0, 1000), (1000, 1001), (1001, 2000)]


def test_rejects_measurement_without_scroller(fixture_shaped) -> None:
    m, _ = fixture_shaped
    bad = Measurement(**{**_fields(m), "continuous": None})
    with pytest.raises(ValueError):
        _assemble(bad)
    h = HeuristicInteraction("drag", "drag", "none", "continuous", 0.7, 0)
    assert cas.continuous_heuristic(m.continuous.scroller).type == h.type  # type: ignore[union-attr]


def _fields(m: Measurement) -> dict[str, Any]:
    return {f: getattr(m, f) for f in m.__dataclass_fields__}


def test_card_scale_from_the_census() -> None:
    """P2c: the census's quadratic scale → ``continuous.card_scale`` (value at the farthest
    measured card, mean magnification over the region, confidence capped at medium)."""
    s = make_series(lambda t: np.full_like(t, 35.0), fps=60.0, dur=6.0, sigma=0.05)
    sa = _analysis(s, pitch=167.8)
    k = 2.48e-6
    sa.card_len_px, sa.card_cross_px, sa.card_zoom_per_px2 = 162.8, 147.2, k
    sa.card_zoom_reach_px, sa.card_zoom_confidence = 261.3, 0.95
    spec = _assemble(_measurement(sa, _probe(6.0, 60.0, False)))
    c = spec.continuous
    assert c is not None and c.card_scale is not None
    cs = c.card_scale
    assert cs.reference_distance_px == 261.3
    assert cs.scale_at_reference == pytest.approx(1.0 + k * 261.3**2, abs=1e-4)
    half = c.region.w / 2.0
    assert cs.mean_scale == pytest.approx(1.0 + k * half * half / 3.0, abs=2e-4)
    assert cs.confidence.value == 0.6 and cs.confidence.band == "medium"
    # rigid cards / no census: omitted
    assert _assemble(_measurement(_analysis(s, pitch=167.8))).continuous.card_scale is None  # type: ignore[union-attr]
    # a reach beyond the region is clamped to its half length
    sa.card_zoom_reach_px = 5000.0
    far = _assemble(_measurement(sa, _probe(6.0, 60.0, False))).continuous
    assert far is not None and far.card_scale is not None
    assert far.card_scale.reference_distance_px == round(half, 1)
