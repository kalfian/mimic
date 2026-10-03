"""Ground truth: internal consistency, round-trip, and agreement with the plan's scenario table."""

from __future__ import annotations

import pytest

from app.config import REPO_ROOT
from app.models.ir import PROPERTY_VALUE_KIND

from . import scenarios as S
from .truth import Truth

SYNTH_DIR = REPO_ROOT / "data" / "synth"
NAMES = [e.name for e in S.SCENARIOS]


@pytest.fixture(scope="module")
def truths() -> dict[str, Truth]:
    return {e.name: S.build_truth(e) for e in S.SCENARIOS}


def test_every_plan_scenario_is_registered() -> None:
    assert {e.scenario for e in S.SCENARIOS} == {f"S{i}" for i in range(1, 13)}
    for suffix in ("vfr", "30fps", "crf28", "mov", "webm"):
        assert f"card_hover_{suffix}" in S.BY_NAME
    assert {"negative_static", "negative_scroll"} <= set(S.BY_NAME)


@pytest.mark.parametrize("name", NAMES)
def test_truth_round_trips(name: str, truths: dict[str, Truth], tmp_path) -> None:
    t = truths[name]
    again = Truth.model_validate_json(t.to_json())
    assert again == t
    assert again.to_json() == t.to_json()
    p = tmp_path / "x.truth.json"
    t.write(p)
    assert Truth.load(p) == t


@pytest.mark.parametrize("name", NAMES)
def test_truth_is_consistent(name: str, truths: dict[str, Truth]) -> None:
    t = truths[name]
    if t.expected_error is not None:
        assert t.interaction is None and not t.transitions
        return
    assert t.interaction is not None
    assert t.interaction.type in t.interaction.accept_types
    seg_ids = {s.id for s in t.segments}
    expected_segs = {
        "forward": {"fwd"}, "forward_reverse": {"fwd", "rev"}, "round_trip": {"rt_in", "rt_out"}
    }[t.interaction.direction]  # fmt: skip
    assert seg_ids == expected_segs
    el_ids = {e.id for e in t.elements}
    onset = {s.id: s.onset_ms for s in t.segments}
    for tr in t.transitions:
        assert tr.element_id in el_ids
        kind = PROPERTY_VALUE_KIND[tr.property]
        assert tr.from_.kind == kind and tr.to.kind == kind
        if kind in ("px", "ratio"):
            assert tr.delta == pytest.approx(tr.to.number - tr.from_.number)  # type: ignore[union-attr]
        else:
            assert tr.delta is None
        assert tr.delay_ms == tr.start_ms - onset[tr.segment_id] >= 0
    for s in t.segments:
        trs = [x for x in t.transitions if x.segment_id == s.id]
        assert s.onset_ms == min(x.start_ms for x in trs)
        assert s.settle_ms == max(x.start_ms + x.duration_ms for x in trs)
    for rel in t.relationships:
        assert all(t.transition(i).segment_id == rel.segment_id for i in rel.transition_ids)
    assert t.interaction.target_element_id in el_ids


def test_s1_matches_plan(truths: dict[str, Truth]) -> None:
    t = truths["card_hover_translate"]
    fwd, rev = t.transitions
    assert (fwd.property, fwd.from_.number, fwd.to.number) == ("translateY", 0.0, -8.0)  # type: ignore[union-attr]
    assert fwd.start_ms == 1000 and fwd.duration_ms == 280 and fwd.easing.keyword == "ease-out"
    assert rev.duration_ms == 220 and rev.easing.keyword == "ease-in-out"
    assert rev.start_ms - (fwd.start_ms + fwd.duration_ms) >= 1200  # hold >= 1.2 s
    card = t.element("card")
    assert (card.bbox_initial.w, card.bbox_initial.h) == (320, 400)
    assert card.bbox_active.y == pytest.approx(card.bbox_initial.y - 8)
    kinds = [e.kind for e in t.cursor.events]
    assert "enter" in kinds and "leave" in kinds
    enter = next(e for e in t.cursor.events if e.kind == "enter")
    leave = next(e for e in t.cursor.events if e.kind == "leave")
    assert enter.t_ms == fwd.start_ms and leave.t_ms == rev.start_ms


def test_s2_compound(truths: dict[str, Truth]) -> None:
    t = truths["card_hover_compound"]
    img = t.element("image")
    assert img.parent_id == "card" and t.element("title").parent_id == "card"
    scale = next(x for x in t.transitions if x.property == "scale" and x.segment_id == "fwd")
    assert scale.delay_ms == 30 and scale.duration_ms == 320
    assert scale.transform_origin == "center"
    assert "relative to parent card" in scale.notes
    color = next(x for x in t.transitions if x.property == "color" and x.segment_id == "fwd")
    assert (color.from_.color, color.to.color) == ("#111111", "#5252FF")  # type: ignore[union-attr]
    shadow = next(x for x in t.transitions if x.property == "box-shadow" and x.segment_id == "fwd")
    assert shadow.to.shadow.blur > shadow.from_.shadow.blur  # type: ignore[union-attr]
    kinds = {(r.segment_id, r.kind) for r in t.relationships}
    assert ("fwd", "delayed") in kinds and ("fwd", "simultaneous") in kinds
    # the scaled image stays clipped by the card
    assert (
        img.bbox_active.w == pytest.approx(320)
        and img.bbox_active.y >= t.element("card").bbox_active.y
    )


def test_s8_stagger_interval(truths: dict[str, Truth]) -> None:
    t = truths["stagger_list"]
    for rel in t.relationships:
        assert rel.kind == "stagger" and rel.interval_ms == 60
        starts = [t.transition(i).start_ms for i in rel.transition_ids]
        assert [b - a for a, b in zip(starts, starts[1:], strict=False)] == [60, 60, 60]


def test_retina_truth_stays_in_css_px(truths: dict[str, Truth]) -> None:
    a, b = truths["card_hover_translate"], truths["card_hover_retina"]
    assert (b.video.width, b.video.height, b.video.pixel_ratio) == (2560, 1600, 2)
    assert a.elements == b.elements and a.transitions == b.transitions


def test_generated_files_match_definitions() -> None:
    """If ``make synth`` has run, the files on disk equal a fresh derivation (minus probe)."""
    files = sorted(SYNTH_DIR.glob("*.truth.json"))
    if not files:
        pytest.skip("no data/synth truth files (run `make synth`)")
    for f in files:
        disk = Truth.load(f)
        if disk.name not in S.BY_NAME:
            continue
        fresh = S.build_truth(S.get(disk.name))
        assert disk.video.probe is not None
        disk.video.probe = None
        assert disk == fresh, f"{f.name} is stale; re-run `make synth`"
