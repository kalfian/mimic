"""Ground truth: internal consistency, round-trip, and agreement with the plan's scenario table."""

from __future__ import annotations

import numpy as np
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
    assert {e.scenario for e in S.TRANSITION_SCENARIOS} == {f"S{i}" for i in range(1, 13)}
    assert len(S.TRANSITION_SCENARIOS) == 17
    for suffix in ("vfr", "30fps", "crf28", "mov", "webm"):
        assert f"card_hover_{suffix}" in S.BY_NAME
    assert {"negative_static", "negative_scroll"} <= set(S.BY_NAME)
    # PLAN-continuous §8.2
    assert [e.scenario for e in S.CONTINUOUS_SCENARIOS] == [f"C{i}" for i in range(1, 15)] + ["N1"]
    assert [e.name for e in S.CONTINUOUS_SCENARIOS] == [
        "marquee_autoplay", "marquee_fast_loop", "marquee_hover_pause", "carousel_drag_inertia",
        "carousel_drag_inertia_no_cursor", "carousel_drag_inertia_30fps",
        "carousel_drag_inertia_vfr", "carousel_drag_snap", "carousel_fling_fast_crf28",
        "ticker_vertical_autoplay", "hover_with_ambient_marquee", "carousel_drag_inertia_grid75",
        "carousel_dark_edge_zoom", "carousel_flat_held_stop", "negative_ambient_pulse",
    ]  # fmt: skip
    assert len(S.BY_NAME) == len(S.SCENARIOS)


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
        "forward": {"fwd"}, "forward_reverse": {"fwd", "rev"}, "round_trip": {"rt_in", "rt_out"},
        "continuous": set(),
    }[t.interaction.direction]  # fmt: skip
    assert (t.continuous is not None) == (t.mode == "continuous")
    if t.continuous is not None:
        assert t.interaction.direction == "continuous" and not t.transitions
        assert t.suite == "continuous"
        assert t.interaction.target_element_id == t.continuous.element_id
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


# ---- PLAN-continuous truth --------------------------------------------------------------------

CONT = [e.name for e in S.CONTINUOUS_SCENARIOS if e.name.startswith(("marquee", "carousel",
                                                                     "ticker"))]  # fmt: skip


@pytest.mark.parametrize("name", CONT)
def test_continuous_truth_is_consistent(name: str, truths: dict[str, Truth]) -> None:
    t = truths[name]
    tc = t.continuous
    assert tc is not None and t.mode == "continuous" and t.suite == "continuous"
    start, end = tc.span_ms
    assert start == 0 and end == t.video.duration_ms
    ph = tc.phases
    assert ph[0].start_ms == start and ph[-1].end_ms == end
    for a, b in zip(ph, ph[1:], strict=False):
        assert a.end_ms == b.start_ms and a.kind != b.kind  # contiguous, merged
    rows = {r[0]: r for r in tc.profile}
    for p in ph:
        if p.start_ms in rows and p.end_ms in rows:  # profile rows every 20 ms
            moved = rows[p.end_ms][1] - rows[p.start_ms][1]
            assert moved == pytest.approx(p.displacement_px, abs=1e-3)
        assert abs(p.v_peak_px_s) >= max(abs(p.v_start_px_s), abs(p.v_end_px_s)) - 1e-6
    held = 0
    for a, b in zip(ph, ph[1:], strict=False):
        if a.kind == "drag":
            assert a.v_start_px_s == 0.0 and a.pointer_ratio == 1.0  # press stops autoplay
            if a.v_end_px_s == 0.0:  # stopped with the pointer while pressed (C14): it rests
                assert b.kind == "paused"
                held += 1
                continue
            assert b.kind in ("inertia", "snap", "stop")
            if b.kind == "inertia":
                assert b.v0_px_s == pytest.approx(a.v_end_px_s)  # release velocity carries over
        if a.kind == "resume":
            assert b.kind == "autoplay" and a.v_end_px_s == tc.autoplay_velocity_px_s
    assert tc.drag_count == sum(p.kind == "drag" for p in ph)
    assert len(tc.release_speeds_px_s) == tc.drag_count - held
    if tc.zoom is None:
        strip = S.get(name).build().continuous.strip  # type: ignore[union-attr]
        assert (tc.card.w, tc.card.h, tc.gap_px) == (strip.card_w, strip.card_h, strip.gap)
        if name != "carousel_flat_held_stop":  # §8.2 strips; C14 uses the recording's cards
            assert tc.card.w == 200 and tc.card.h == 160 and tc.gap_px == 16
            assert tc.pitch_px == (216 if tc.axis == "x" else 176)
        along = tc.card.w if tc.axis == "x" else tc.card.h
        assert tc.pitch_px == pytest.approx(along + tc.gap_px)
        assert tc.loop.period_px == tc.card.count * tc.pitch_px
    else:  # C13: cards scale with position; card / gap / pitch at the viewport centre
        strip = S.get(name).build().continuous.strip  # type: ignore[union-attr]
        assert (tc.card.w, tc.card.h) == (strip.card_w, strip.card_h)
        assert tc.pitch_px == pytest.approx(tc.card.w + tc.gap_px, abs=1e-3)
        assert tc.loop.period_px == pytest.approx(strip.period * tc.zoom.speed_scale, abs=1e-3)
    scroller = t.element(tc.element_id)
    assert (scroller.kind, scroller.role) == ("scroller", "scroller")
    assert scroller.bbox_initial == tc.region
    assert scroller.appearance is not None and scroller.appearance.background_color is not None


def test_c1_c2_loop_observability(truths: dict[str, Truth]) -> None:
    c1, c2 = truths["marquee_autoplay"].continuous, truths["marquee_fast_loop"].continuous
    assert c1 is not None and c2 is not None
    assert (c1.autoplay_velocity_px_s, c1.autoplay_direction) == (-60.0, "left")
    assert c1.loop.period_px == 1728 and not c1.loop.observable  # travel 480 px
    assert (c2.autoplay_velocity_px_s, c2.loop.period_px) == (-240.0, 864)
    assert c2.loop.observable and c2.loop.duration_ms == 3600.0
    assert [p.kind for p in c1.phases] == ["autoplay"]


@pytest.mark.parametrize("name", ["marquee_hover_pause", "ticker_vertical_autoplay"])
def test_hover_pause_timings(name: str, truths: dict[str, Truth]) -> None:
    t = truths[name]
    tc = t.continuous
    assert tc is not None
    assert [p.kind for p in tc.phases] == ["autoplay", "decelerate", "paused", "resume",
                                           "autoplay"]  # fmt: skip
    _, dec, hold, res, _ = tc.phases
    enter = next(e for e in t.cursor.events if e.kind == "enter")
    leave = next(e for e in t.cursor.events if e.kind == "leave")
    assert enter.t_ms == dec.start_ms == 2000 and dec.ramp_ms == 400
    assert dec.easing is not None and dec.easing.keyword == "ease-out"
    assert leave.t_ms in (4000, 4001) and res.start_ms - leave.t_ms == 300
    assert res.ramp_ms == 600 and res.easing is not None and res.easing.keyword == "ease-in"
    assert tc.pause_on == "hover" and tc.pause_decel_ms == 400
    assert tc.resume_delay_after_rest_ms == res.start_ms - dec.end_ms
    assert tc.resume_delay_after_release_ms is None and tc.snap_kind is None
    if name == "ticker_vertical_autoplay":
        assert (tc.axis, tc.autoplay_direction, tc.autoplay_velocity_px_s) == ("y", "up", -30.0)
    else:
        assert (tc.axis, tc.autoplay_direction, tc.autoplay_velocity_px_s) == ("x", "right", 40.0)


def test_c4_carousel_values(truths: dict[str, Truth]) -> None:
    t = truths["carousel_drag_inertia"]
    tc = t.continuous
    assert tc is not None
    assert [p.kind for p in tc.phases] == [
        "autoplay", "drag", "inertia", "paused", "resume", "autoplay",
        "drag", "inertia", "paused", "resume", "autoplay",
    ]  # fmt: skip
    drags = [p for p in tc.phases if p.kind == "drag"]
    assert drags[0].start_ms == 2000 and drags[0].end_ms - drags[0].start_ms == 450
    assert drags[0].displacement_px == pytest.approx(-400.0)
    assert [p.v0_px_s for p in tc.phases if p.kind == "inertia"] == [-1100.0, 1000.0]
    assert tc.inertia_tau_ms == 200.0 and tc.release_speeds_px_s == [1100.0, 1000.0]
    assert tc.drag_peak_speed_px_s == 1100.0
    assert tc.drag_follows_pointer is True and tc.drag_pointer_ratio == 1.0
    assert tc.resume_delay_after_rest_ms == 1000 and tc.resume_ramp_ms == 750
    assert tc.pause_on == "press" and tc.pause_on_accept == ["press"]
    assert tc.pause_decel_ms is None  # the press stops autoplay instantly
    downs = [e.t_ms for e in t.cursor.events if e.kind == "mousedown"]
    ups = [e.t_ms for e in t.cursor.events if e.kind == "mouseup"]
    assert downs == [p.start_ms for p in drags] and ups == [p.end_ms for p in drags]


def test_c5_to_c7_share_c4_motion(truths: dict[str, Truth]) -> None:
    c4 = truths["carousel_drag_inertia"].continuous
    for name in ("carousel_drag_inertia_no_cursor", "carousel_drag_inertia_30fps",
                 "carousel_drag_inertia_vfr"):  # fmt: skip
        tc = truths[name].continuous
        assert tc is not None and c4 is not None
        assert tc.phases == c4.phases and tc.profile == c4.profile
    c5 = truths["carousel_drag_inertia_no_cursor"]
    assert not c5.cursor.visible and c5.cursor.events == []
    assert c5.continuous.pause_on_accept == ["unknown"]  # type: ignore[union-attr]
    assert c5.continuous.drag_follows_pointer is None  # type: ignore[union-attr]
    assert truths["carousel_drag_inertia_30fps"].video.fps == 30
    assert truths["carousel_drag_inertia_vfr"].video.vfr


def test_c8_snaps_rest_on_the_card_grid(truths: dict[str, Truth]) -> None:
    tc = truths["carousel_drag_snap"].continuous
    assert tc is not None and tc.snap_kind == "grid"
    ends = np.cumsum([p.displacement_px for p in tc.phases])  # x(0) = 0
    snaps = [(p, ends[i]) for i, p in enumerate(tc.phases) if p.kind == "snap"]
    assert len(snaps) == 2
    for p, rest in snaps:
        assert p.end_ms - p.start_ms == 300 and p.easing.keyword == "ease-out"  # type: ignore[union-attr]
        assert p.snap_step_px == 216
        assert rest % 216 == pytest.approx(0.0, abs=1e-3) or rest % 216 == pytest.approx(216.0)
        assert abs(p.snap_distance_px) >= 54  # type: ignore[arg-type]
        assert np.sign(p.snap_distance_px) == np.sign(p.v_start_px_s)
    for name in CONT:
        if name != "carousel_drag_snap":
            assert truths[name].continuous.snap_kind is None  # type: ignore[union-attr]


def test_c9_fling_stays_below_half_a_pitch_per_frame(truths: dict[str, Truth]) -> None:
    t = truths["carousel_fling_fast_crf28"]
    tc = t.continuous
    assert tc is not None and (t.video.fps, t.video.crf) == (30.0, 28)
    assert tc.drag_peak_speed_px_s == 2500.0 and tc.inertia_tau_ms == 250.0
    assert tc.drag_peak_speed_px_s / t.video.fps < tc.pitch_px / 2


def test_c11_is_s1_plus_an_ambient_marquee(truths: dict[str, Truth]) -> None:
    s1, c11 = truths["card_hover_translate"], truths["hover_with_ambient_marquee"]
    assert (c11.suite, c11.mode, c11.continuous) == ("continuous", "transition", None)
    assert c11.expected_warnings == ["ambient_motion_masked"]
    assert c11.elements == s1.elements and c11.transitions == s1.transitions
    assert c11.segments == s1.segments and c11.cursor == s1.cursor
    (amb,) = c11.ambient
    assert (amb.axis, amb.autoplay_velocity_px_s) == ("x", -50.0)
    assert [p.kind for p in amb.phases] == ["autoplay"] and amb.phases[0].start_ms == 0
    card = s1.element("card").bbox_initial
    assert amb.region.y > card.y + card.h + 16  # well clear of the hovered card


def test_n1_negative_ambient_pulse(truths: dict[str, Truth]) -> None:
    t = truths["negative_ambient_pulse"]
    assert (t.expected_error, t.suite) == ("continuous_motion_unsupported", "continuous")
    assert t.interaction is None and t.continuous is None
