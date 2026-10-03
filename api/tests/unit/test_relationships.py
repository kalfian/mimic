"""Track M2 — relationships.py on synthetic t0/D tables (PLAN §6.9)."""

from __future__ import annotations

import numpy as np
import pytest

from app.models.ir import PxValue, Relationship
from app.models.measure import FittedTransition, PropertySeries
from app.pipeline import relationships as rl
from app.pipeline.confidence import make_transition_confidence
from app.pipeline.easing import make_easing

T = rl.TimedTransition
RES_60 = 1.0 / 60.0


def kinds(rels: list[Relationship]) -> list[tuple[str, list[str], int | None, int | None]]:
    return [(r.kind, r.transition_ids, r.offset_ms, r.interval_ms) for r in rels]


def test_tau() -> None:
    assert rl.tau_s(RES_60) == pytest.approx(0.020)
    assert rl.tau_s(1 / 30) == pytest.approx(1 / 30)


def test_card_hover_compound_like_fixture() -> None:
    """S2-like: card translate + shadow + title color together, image scale +30 ms, arrow +80 ms."""
    fwd = [
        T("t1", "fwd", "e1", "translateY", 1.180, 0.280),
        T("t2", "fwd", "e1", "box-shadow", 1.181, 0.280),
        T("t3", "fwd", "e2", "scale", 1.210, 0.320),
        T("t4", "fwd", "e3", "color", 1.178, 0.200),
        T("t5", "fwd", "e4", "translateX", 1.260, 0.200),
        T("t6", "fwd", "e4", "opacity", 1.262, 0.200),
    ]
    rels = rl.segment_relationships(fwd, target_element_id="e1", timing_resolution_s=RES_60)
    assert kinds(rels) == [
        ("simultaneous", ["t1", "t4", "t2"], 0, None),
        ("delayed", ["t1", "t3"], 30, None),
        ("delayed", ["t1", "t5", "t6"], 81, None),
    ]


def test_primary_is_target_earliest_with_fallback() -> None:
    ts = [T("t1", "fwd", "e2", "scale", 1.00, 0.3), T("t2", "fwd", "e1", "translateY", 1.05, 0.3)]
    assert rl.pick_primary(ts, "e1").id == "t2"
    assert rl.pick_primary(ts, "e9").id == "t1"
    assert rl.pick_primary(ts, None).id == "t1"
    rels = rl.segment_relationships(ts, target_element_id="e1", timing_resolution_s=RES_60)
    # earlier than the primary -> negative offset is still "delayed" relative to it
    assert kinds(rels) == [("delayed", ["t2", "t1"], -50, None)]


def test_sequential() -> None:
    ts = [
        T("t1", "fwd", "e1", "opacity", 1.0, 0.20),
        T("t2", "fwd", "e2", "translateY", 1.19, 0.20),  # starts as t1 ends (within τ)
        T("t3", "fwd", "e3", "translateY", 1.10, 0.20),  # overlaps -> delayed
    ]
    rels = rl.segment_relationships(ts, target_element_id="e1", timing_resolution_s=RES_60)
    assert kinds(rels) == [
        ("delayed", ["t1", "t3"], 100, None),
        ("sequential", ["t1", "t2"], 190, None),
    ]


def _stagger_list(offsets_ms: list[float], areas: list[float | None]) -> list[rl.TimedTransition]:
    out = []
    for i, (off, area) in enumerate(zip(offsets_ms, areas, strict=True)):
        t0 = 1.0 + off / 1000.0
        e = f"e{i + 1}"
        out.append(T(f"t{2 * i + 1}", "fwd", e, "opacity", t0, 0.2, area, "list_item"))
        out.append(T(f"t{2 * i + 2}", "fwd", e, "translateY", t0, 0.2, area, "list_item"))
    return out


def test_stagger_list() -> None:
    """S8: 4 items fade + translateY, 200 ms each, 60 ms stagger."""
    ts = _stagger_list([0, 61, 119, 181], [3000, 3100, 2950, 3050])
    rels = rl.segment_relationships(ts, target_element_id=None, timing_resolution_s=RES_60)
    assert kinds(rels) == [
        ("simultaneous", ["t1", "t2"], 0, None),
        ("stagger", ["t1", "t3", "t5", "t7"], 0, 60),
        ("stagger", ["t2", "t4", "t6", "t8"], 0, 60),
    ]
    stagger = [r for r in rels if r.kind == "stagger"][0]
    assert abs(stagger.interval_ms - 60) <= 15  # type: ignore[operator]


def test_no_stagger_when_irregular_dissimilar_or_too_few() -> None:
    irregular = _stagger_list([0, 20, 150, 170], [3000] * 4)
    rels = rl.segment_relationships(irregular, target_element_id=None, timing_resolution_s=RES_60)
    assert all(r.kind != "stagger" for r in rels)
    dissimilar = _stagger_list([0, 60, 120, 180], [3000, 9000, 3000, 9000])
    rels = rl.segment_relationships(dissimilar, target_element_id=None, timing_resolution_s=RES_60)
    assert all(r.kind != "stagger" for r in rels)
    two = _stagger_list([0, 60], [3000, 3000])
    rels = rl.segment_relationships(two, target_element_id=None, timing_resolution_s=RES_60)
    assert all(r.kind != "stagger" for r in rels)
    together = _stagger_list([0, 5, 10, 12], [3000] * 4)  # mean offset < τ -> simultaneous
    rels = rl.segment_relationships(together, target_element_id=None, timing_resolution_s=RES_60)
    assert [r.kind for r in rels] == ["simultaneous"]
    assert len(rels[0].transition_ids) == 8


def test_stagger_with_unknown_areas_uses_role_only() -> None:
    ts = _stagger_list([0, 60, 120], [None, None, None])
    rels = rl.segment_relationships(ts, target_element_id=None, timing_resolution_s=RES_60)
    assert [r.kind for r in rels].count("stagger") == 2


def test_multi_segment_and_validation() -> None:
    ts = [
        T("t1", "fwd", "e1", "translateY", 1.0, 0.28),
        T("t2", "fwd", "e1", "box-shadow", 1.0, 0.28),
        T("t3", "rev", "e1", "translateY", 2.8, 0.22),
        T("t4", "rev", "e1", "box-shadow", 2.81, 0.22),
        T("t5", "rev", "e2", "scale", 2.85, 0.22),
    ]
    rels = rl.relationships(ts, target_element_id="e1", timing_resolution_s=RES_60)
    assert [(r.segment_id, r.kind) for r in rels] == [
        ("fwd", "simultaneous"),
        ("rev", "simultaneous"),
        ("rev", "delayed"),
    ]
    assert (
        rl.segment_relationships(ts[:1], target_element_id="e1", timing_resolution_s=RES_60) == []
    )
    with pytest.raises(ValueError):
        rl.segment_relationships(ts, target_element_id="e1", timing_resolution_s=RES_60)


def test_from_fitted_adapter() -> None:
    series = PropertySeries(
        element_id="e2",
        property="translateY",
        segment="rev",
        times=np.zeros(2),
        values=np.zeros(2),
        quality=np.ones(2),
        v_start=-8,
        v_end=0,
        unit="px",
        from_value=PxValue(number=-8),
        to_value=PxValue(number=0),
    )
    ft = FittedTransition(
        id="t7",
        series=series,
        t0_s=2.84,
        duration_s=0.22,
        easing=make_easing((0.42, 0, 0.58, 1), 0.01),
        confidence=make_transition_confidence(0.9, 0.9, 0.6, cap=0.95),
    )
    tt = rl.from_fitted(ft, area=1200.0, role="card")
    assert (tt.id, tt.segment, tt.element_id, tt.property) == ("t7", "rev", "e2", "translateY")
    assert tt.end_s == pytest.approx(3.06) and tt.area == 1200.0 and tt.role == "card"
