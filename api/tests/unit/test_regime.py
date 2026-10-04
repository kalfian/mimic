"""PLAN-continuous Track R: ambient regions (§3.2), masked scan routing, ``decide_mode`` (§3.3)."""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from app.config import REPO_ROOT, get_settings
from app.core.errors import ErrorCode, PipelineError
from app.models.measure import (
    ActiveSegment,
    AmbientRegion,
    CursorTrack,
    DisplacementSeries,
    PhaseCandidate,
    PrimaryInteraction,
    Rect,
    RegimeReport,
    ScanResult,
    ScrollerAnalysis,
)
from app.pipeline.continuous.regime import (
    ambient_regions,
    apply_ambient_mask,
    cells_rect,
    decide_mode,
    describe_region,
    detect_regime,
    lead_frames,
    merge_row_pieces,
    scan_for_regime,
)
from app.pipeline.params import DEFAULT_PARAMS as P
from app.pipeline.scan import Pass1Accumulator, Pass1Stats, build_scan_result
from tests.unit.m1_helpers import (
    base_canvas,
    composite,
    requires_ffmpeg,
    scale1,
    solid_rect_sprite,
    translate,
    window_frames,
)

FPS = 30.0
FRAME = (1280.0, 800.0)
GRID = (50, 80)  # rows, cols of 16 px cells


def _times(n: int, fps: float = FPS) -> np.ndarray:
    return np.arange(n) / fps


def _grid(n: int = 60) -> np.ndarray:
    return np.zeros((n, *GRID), dtype=bool)


# ---- §3.2 on hand-built activity grids -------------------------------------------------------


def test_static_grid_has_no_ambient() -> None:
    rep = ambient_regions(_grid(), _times(60), FRAME, P)
    assert rep.ambient == [] and not rep.has_ambient
    assert rep.grid_shape == GRID and rep.cell_css == P.continuous.cell_css
    assert rep.lead_window_s == (0.0, pytest.approx(0.6))


def test_strip_active_from_start_is_ambient() -> None:
    act = _grid()
    act[1:, 18:31, 5:53] = True  # rows 18..30, cols 5..52, every frame after the first
    (a,) = ambient_regions(act, _times(60), FRAME, P).ambient
    assert a.rect_css == Rect(80.0, 288.0, 768.0, 208.0)
    assert a.cells == 13 * 48 and a.lead_frac == pytest.approx(1.0)


def test_intermittent_edges_still_ambient() -> None:
    """Slow textured content: each cell active in ~60 % of frames, not always (§2 rejected alt)."""
    rng = np.random.default_rng(3)
    act = _grid()
    act[1:, 20:26, 10:40] = rng.random((59, 6, 30)) < 0.65
    (a,) = ambient_regions(act, _times(60), FRAME, P).ambient
    assert a.rect_css.contained_fraction(cells_rect(20, 10, 25, 39, GRID, FRAME)) == 1.0
    assert a.rect_css.area >= 0.8 * cells_rect(20, 10, 25, 39, GRID, FRAME).area


def test_motion_after_static_lead_is_not_ambient() -> None:
    act = _grid(90)
    act[30:, 18:31, 5:53] = True  # starts at 1.0 s: regime A by definition (§3.2 rule 4)
    assert ambient_regions(act, _times(90), FRAME, P).ambient == []


def test_sparse_lead_activity_is_not_ambient() -> None:
    act = _grid()
    act[1::3, 18:31, 5:53] = True  # one frame in three < ambient_lead_frac
    assert ambient_regions(act, _times(60), FRAME, P).ambient == []


def test_cursor_cells_are_removed() -> None:
    act = _grid()
    act[1:, 10:12, 10:12] = True  # a jittering pointer covering 2x2 cells
    assert len(ambient_regions(act, _times(60), FRAME, P).ambient) == 1
    boxes = [None] + [Rect(160.0, 160.0, 32.0, 32.0)] * 59
    assert ambient_regions(act, _times(60), FRAME, P, cursor_boxes=boxes).ambient == []


def test_cursor_box_inside_moving_content_is_kept() -> None:
    """A "cursor" over an active strip (pointer over the scroller, or drifting content that the
    tracker mistook for one) must not punch a hole into the ambient region."""
    act = _grid()
    act[1:, 18:31, 5:53] = True
    boxes = [None] + [Rect(16.0 * (10 + i // 3), 320.0, 40.0, 40.0) for i in range(59)]
    (a,) = ambient_regions(act, _times(60), FRAME, P, cursor_boxes=boxes).ambient
    assert a.rect_css == Rect(80.0, 288.0, 768.0, 208.0) and a.cells == 13 * 48


def test_single_cell_is_dropped_and_close_bridges_gaps() -> None:
    act = _grid()
    act[1:, 5, 5] = True  # isolated cell: < ambient_min_cells
    act[1:, 30:33, 10:20] = True
    act[1:, 30:33, 21:30] = True  # one-cell gap → bridged by the 3x3 close
    (a,) = ambient_regions(act, _times(60), FRAME, P).ambient
    assert a.rect_css == cells_rect(30, 10, 32, 29, GRID, FRAME)
    assert a.cells == 3 * 19


def test_regions_split_sorted_and_capped() -> None:
    act = _grid()
    act[1:, 2:4, 2:10] = True  # 16 cells
    act[1:, 10:14, 2:30] = True  # 112 cells
    act[1:, 20:22, 2:40] = True  # 76 cells
    act[1:, 30:31, 2:12] = True  # 10 cells → beyond max_ambient_regions
    rep = ambient_regions(act, _times(60), FRAME, P)
    assert [a.cells for a in rep.ambient] == [112, 76, 16]
    assert len(rep.ambient) == P.continuous.max_ambient_regions


def test_lead_window_bounds() -> None:
    assert lead_frames(_times(60), P) == list(range(1, 19))  # 30 fps: (0, 0.6] without frame 0
    assert lead_frames(_times(60, 10.0), P) == list(range(1, 11))  # extended to min frames
    assert lead_frames(_times(8), P) == []  # too short to call anything ambient
    act = np.ones((8, *GRID), dtype=bool)
    assert ambient_regions(act, _times(8), FRAME, P).ambient == []


def test_non_divisible_frame_cells_tile_the_frame() -> None:
    frame = (904.0, 786.0)
    grid = (50, 57)
    r = cells_rect(0, 0, grid[0] - 1, grid[1] - 1, grid, frame)
    assert r == Rect(0.0, 0.0, 904.0, 786.0)


# ---- §3.2 on pass-1 stats from rendered frames -----------------------------------------------

STRIP_XY, STRIP_WH = (40, 120), (240, 64)


def _marquee_frames(n: int, start_s: float = 0.0, v: float = 40.0) -> list[np.ndarray]:
    # dense content (card-like stripes of random gray, 6 px wide) like a real carousel strip
    rng = np.random.default_rng(5)
    cols = np.repeat(rng.integers(40, 220, (STRIP_WH[0] + 400) // 6 + 1), 6)[: STRIP_WH[0] + 400]
    tex = np.repeat(np.tile(cols.astype(np.uint8), (STRIP_WH[1], 1))[..., None], 3, axis=2)
    card = solid_rect_sprite(48, 32, (90, 90, 200))
    out = []
    for i in range(n):
        t = i / FPS
        img = base_canvas(320, 240)
        off = int(round(max(0.0, t - start_s) * v))
        x, y = STRIP_XY
        img[y : y + STRIP_WH[1], x : x + STRIP_WH[0]] = tex[:, off : off + STRIP_WH[0]]
        composite(img, card, translate(200, 196 + (4 if 1.5 <= t < 1.8 else 0)))
        out.append(img)
    return out


def _accumulate(frames: list[np.ndarray]) -> Pass1Stats:
    h, w = frames[0].shape[:2]
    acc = Pass1Accumulator(params=P, scale=scale1(), image_scale=1.0, frame_css=(w, h))
    for i, f in enumerate(frames):
        acc.add(i / FPS, f.mean(axis=2).astype(np.uint8))
    return acc.finish()


def test_detect_regime_finds_the_marquee() -> None:
    stats = _accumulate(_marquee_frames(75))
    rep = detect_regime(stats, P)
    (a,) = rep.ambient
    strip = Rect(*STRIP_XY, *STRIP_WH)
    assert a.rect_css.iou(strip) >= 0.7
    assert a.rect_css.contained_fraction(strip.padded(16)) == 1.0
    assert rep.grid_shape == (15, 20)
    # the masked scan then finds the card's move (1.5 s) and return (1.8 s) outside the strip
    res = scan_for_regime(stats, scale1(), P, rep)
    assert isinstance(res, ScanResult) and len(res.segments) == 2
    assert all(s.roi_css.intersection(strip) is None for s in res.segments)
    assert 1.4 <= res.segments[0].start_s <= 1.55 and 1.7 <= res.segments[1].start_s <= 1.85
    assert res.primary.reverse is not None


def test_detect_regime_ignores_marquee_starting_later() -> None:
    stats = _accumulate(_marquee_frames(75, start_s=1.0))
    assert detect_regime(stats, P).ambient == []


def test_detect_regime_without_activity() -> None:
    stats = _accumulate(_marquee_frames(20))
    stats.activity = None
    rep = detect_regime(stats, P)
    assert rep.ambient == [] and rep.grid_shape == (0, 0)


def test_scan_for_regime_returns_errors_and_matches_default_without_ambient() -> None:
    stats = _accumulate(_marquee_frames(75, start_s=1.0))
    rep = detect_regime(stats, P)
    res = scan_for_regime(stats, scale1(), P, rep)
    expected = None
    try:
        expected = build_scan_result(stats, scale1(), P)
    except PipelineError as exc:
        assert isinstance(res, PipelineError) and res.code == exc.code
    if expected is not None:
        assert isinstance(res, ScanResult) and np.array_equal(res.energy, expected.energy)
    static = _accumulate([base_canvas(320, 240)] * 30)
    err = scan_for_regime(static, scale1(), P, detect_regime(static, P))
    assert isinstance(err, PipelineError) and err.code == ErrorCode.NO_MOTION_DETECTED


# ---- all existing synthetic scenarios stay on the transition path ----------------------------

SYNTH = REPO_ROOT / "data" / "synth"


def _transition_videos() -> list[Path]:
    """The 17 pre-continuous scenarios (S1–S12 + variants): truth ``suite != "continuous"``.
    The PLAN-continuous C/N videos have ambient motion by design and are excluded."""
    out: list[Path] = []
    for truth in sorted(SYNTH.glob("*.truth.json")):
        data = json.loads(truth.read_text(encoding="utf-8"))
        if data.get("suite") == "continuous":
            continue
        video = SYNTH / data["video"]["file"]
        if video.exists():
            out.append(video)
    return out


VIDEOS = _transition_videos()


def _synth_regime(video: Path) -> tuple[str, RegimeReport]:
    from app.pipeline.decode import iter_pass1, pass1_geometry
    from app.pipeline.probe import probe, resolve_scale

    settings = get_settings()
    info = probe(video, settings, P)
    scale, _ = resolve_scale(info, "auto", P)
    g = pass1_geometry(info, P)
    acc = Pass1Accumulator(params=P, scale=scale, image_scale=g.image_scale,
                           frame_css=(info.width / scale.pixel_ratio,
                                      info.height / scale.pixel_ratio))  # fmt: skip
    for t, gray in iter_pass1(video, info, P, settings.ffmpeg_bin):
        acc.add(t, gray)
    return video.name, detect_regime(acc.finish(), P)


@requires_ffmpeg
@pytest.mark.skipif(not VIDEOS, reason="no synthetic videos; run `make synth` first")
def test_no_ambient_regions_in_existing_scenarios() -> None:
    """§9 Track R DoD: every pre-continuous scenario keeps the unchanged transition path."""
    names = {v.stem for v in VIDEOS}
    assert len(names) == 17
    with ThreadPoolExecutor(max_workers=min(4, os.cpu_count() or 1)) as ex:
        found = {name: rep.ambient for name, rep in ex.map(_synth_regime, VIDEOS)}
    assert {k: v for k, v in found.items() if v} == {}
    assert len(found) == len(names)


# ---- §3.4 pass-2 masks -----------------------------------------------------------------------


def test_apply_ambient_mask_ors_into_cursor_masks() -> None:
    frames = [base_canvas(120, 80) for _ in range(3)]
    wf = window_frames(frames, FPS)
    cm = np.zeros((80, 120), np.uint8)
    cm[0:5, 0:5] = 255
    wf.cursor_masks[1] = cm
    region = AmbientRegion(Rect(60.0, 20.0, 30.0, 20.0), cells=4, lead_frac=1.0)
    out = apply_ambient_mask([wf], [region], P)
    assert out[0] is wf
    pad = int(P.continuous.ambient_mask_dilate_css)
    for m in wf.cursor_masks:
        assert m is not None and m.dtype == np.uint8
        assert (m[20 - pad : 40 + pad, 60 - pad : 90 + pad] == 255).all()
        assert m[70, 10] == 0
    assert wf.cursor_masks[1][2, 2] == 255  # cursor pixels kept
    assert cm[30, 70] == 0  # the original mask array is not mutated


def test_apply_ambient_mask_outside_crop_or_empty_is_noop() -> None:
    wf = window_frames([base_canvas(120, 80) for _ in range(2)], FPS)
    apply_ambient_mask([wf], [Rect(500.0, 500.0, 10.0, 10.0)], P)
    assert wf.cursor_masks == [None, None]
    apply_ambient_mask([wf], [], P)
    assert wf.cursor_masks == [None, None]


# ---- §3.3 decide_mode table ------------------------------------------------------------------

BAND = Rect(89.0, 301.0, 746.0, 194.0)
OTHER = Rect(100.0, 600.0, 400.0, 100.0)
AMB = AmbientRegion(BAND, cells=560, lead_frac=0.9)
AMB2 = AmbientRegion(OTHER, cells=150, lead_frac=0.8)
REGIME = RegimeReport([AMB, AMB2], 16.0, GRID, (0.0, 0.6))
NO_AMBIENT = RegimeReport([], 16.0, GRID, (0.0, 0.6))


def _scan(n_segments: int = 1) -> ScanResult:
    seg = ActiveSegment(1.0, 1.3, Rect(900.0, 100.0, 120.0, 90.0), 5000.0)
    return ScanResult(
        energy=np.zeros(10),
        frame_times=np.arange(10) / FPS,
        segments=[seg] * n_segments,
        stable=[],
        thumbs=np.zeros((10, 4, 4), np.uint8),
        primary=PrimaryInteraction(forward=seg),
        cursor=CursorTrack(visible=False),
    )


def _phase(kind: str) -> PhaseCandidate:
    return PhaseCandidate(kind=kind, start_s=0.0, end_s=1.0, v_start=0.0, v_end=0.0,  # type: ignore[arg-type]
                          v_peak=0.0, displacement=0.0)  # fmt: skip


def _scroller(
    region: AmbientRegion = AMB,
    kinds: tuple[str, ...] = ("autoplay",),
    v: float | None = 39.0,
    axis: str = "x",
    box: Rect | None = None,
) -> ScrollerAnalysis:
    series = None
    if box is not None:
        series = DisplacementSeries(np.zeros(2), np.zeros(2), np.ones(2), axis, box, 0.7, 0.02)  # type: ignore[arg-type]
    return ScrollerAnalysis(region=region, is_scroller=True, axis=axis, coherence=0.9,  # type: ignore[arg-type]
                            series=series, autoplay_velocity=v,
                            phases=[_phase(k) for k in kinds])  # fmt: skip


def _not_scroller(region: AmbientRegion = AMB, reason: str | None = "two_axis") -> ScrollerAnalysis:
    return ScrollerAnalysis(region=region, is_scroller=False, reason=reason)  # type: ignore[arg-type]


NO_RUNS = PipelineError(ErrorCode.NO_MOTION_DETECTED)
NO_STABLE = PipelineError(ErrorCode.NO_STABLE_STATE)


def _codes(d) -> list[str]:
    return [w.code for w in d.warnings]


def test_no_ambient_is_the_unchanged_transition_path() -> None:
    scan = _scan()
    d = decide_mode(NO_AMBIENT, [], scan, FRAME, P)
    assert d.mode == "transition" and d.scan is scan and d.warnings == [] and d.masked == []


def test_no_ambient_and_no_runs_is_no_motion_detected() -> None:
    with pytest.raises(PipelineError) as ei:
        decide_mode(NO_AMBIENT, [], NO_RUNS, FRAME, P)
    assert ei.value is NO_RUNS
    with pytest.raises(PipelineError) as ei:
        decide_mode(NO_AMBIENT, [], NO_STABLE, FRAME, P)
    assert ei.value.code == ErrorCode.NO_STABLE_STATE


@pytest.mark.parametrize(
    "box",
    [Rect(0.0, 0.0, 1280.0, 520.0), Rect(10.0, 10.0, 1160.0, 730.0)],  # > 60 % area; ≥ 90 % both
)
def test_page_sized_scroller_is_unsupported_motion(box: Rect) -> None:
    for scan in (_scan(), NO_RUNS):
        with pytest.raises(PipelineError) as ei:
            decide_mode(REGIME, [_scroller(box=box, kinds=("drag",))], scan, FRAME, P)
        assert ei.value.code == ErrorCode.UNSUPPORTED_MOTION


def test_interacted_scroller_is_continuous_largest_wins() -> None:
    big = _scroller(kinds=("autoplay", "decelerate", "drag", "inertia"))
    small = _scroller(region=AMB2, kinds=("autoplay", "drag"))
    d = decide_mode(REGIME, [small, big], NO_RUNS, FRAME, P)
    assert d.mode == "continuous" and d.scroller is big and d.scan is None
    assert d.extra_scrollers == 1 and _codes(d) == ["extra_scrollers_ignored"]
    assert d.masked == [AMB, AMB2]


@pytest.mark.parametrize(
    "kind", ["decelerate", "paused", "drag", "inertia", "snap", "stop", "resume"]
)
def test_every_interaction_phase_counts(kind: str) -> None:
    d = decide_mode(REGIME, [_scroller(kinds=("autoplay", kind))], _scan(), FRAME, P)
    assert d.mode == "continuous"


def test_interacted_scroller_with_masked_runs_ignores_them() -> None:
    scan = _scan(2)
    d = decide_mode(REGIME, [_scroller(kinds=("drag",)), _not_scroller(AMB2)], scan, FRAME, P)
    assert d.mode == "continuous" and d.scan is scan and d.extra_scrollers == 0
    assert _codes(d) == ["extra_segments_ignored"]
    assert d.warnings[0].message.startswith("2 other motion segment(s) outside the scroller")


def test_interacted_scroller_beats_larger_autoplay_only_scroller() -> None:
    marquee = _scroller(kinds=("autoplay",))
    dragged = _scroller(region=AMB2, kinds=("drag", "inertia"))
    d = decide_mode(REGIME, [marquee, dragged], NO_RUNS, FRAME, P)
    assert d.scroller is dragged and d.extra_scrollers == 1


def test_autoplay_only_with_masked_runs_is_masked_transition() -> None:
    scan = _scan()
    d = decide_mode(REGIME, [_scroller(v=-123.0), _not_scroller(AMB2)], scan, FRAME, P)
    assert d.mode == "transition" and d.scan is scan and d.scroller is None
    assert _codes(d) == ["ambient_motion_masked"]
    msg = d.warnings[0].message
    assert d.warnings[0].severity == "info"
    assert describe_region(BAND) in msg and describe_region(OTHER) in msg
    assert "moving left at about 120 px/s" in msg and "Parts of the page" in msg


def test_autoplay_only_without_runs_is_continuous_marquee() -> None:
    a, b = _scroller(), _scroller(region=AMB2, v=12.0)
    d = decide_mode(REGIME, [b, a], NO_RUNS, FRAME, P)
    assert d.mode == "continuous" and d.scroller is a and d.scan is None
    assert _codes(d) == ["extra_scrollers_ignored"]


def test_autoplay_only_with_unusable_runs_falls_back_to_marquee() -> None:
    d = decide_mode(REGIME, [_scroller()], NO_STABLE, FRAME, P)
    assert d.mode == "continuous" and d.scan is None
    assert _codes(d) == ["extra_segments_ignored"]
    assert d.warnings[0].message.startswith("Other motion outside the scroller")


def test_no_scroller_with_masked_runs_is_masked_transition() -> None:
    one = RegimeReport([AMB], 16.0, GRID, (0.0, 0.6))
    d = decide_mode(one, [_not_scroller()], _scan(), FRAME, P)
    assert d.mode == "transition" and _codes(d) == ["ambient_motion_masked"]
    msg = d.warnings[0].message
    assert msg.startswith(f"Part of the page ({describe_region(BAND)})") and "px/s" not in msg


def test_no_scroller_with_unusable_runs_raises_that_error() -> None:
    with pytest.raises(PipelineError) as ei:
        decide_mode(REGIME, [_not_scroller()], NO_STABLE, FRAME, P)
    assert ei.value is NO_STABLE


@pytest.mark.parametrize(
    ("reason", "detail"),
    [
        ("two_axis", "not as a single horizontal or vertical scroller"),
        ("untrackable", "could not be tracked"),
        ("opposing_unsplit", "several scrollers moving in different directions"),
        (None, "not as a single horizontal or vertical scroller"),
    ],
)
def test_no_scroller_no_runs_is_continuous_motion_unsupported(reason, detail) -> None:
    with pytest.raises(PipelineError) as ei:
        decide_mode(REGIME, [_not_scroller(reason=reason), _not_scroller(AMB2)], NO_RUNS, FRAME, P)
    e = ei.value
    assert e.code == ErrorCode.CONTINUOUS_MOTION_UNSUPPORTED and e.http_status == 422
    assert "(around x 89, y 301, 746×194 px)" in e.message and detail in e.message


def test_unanalysed_ambient_without_runs_names_the_largest_region() -> None:
    with pytest.raises(PipelineError) as ei:
        decide_mode(REGIME, [], NO_RUNS, FRAME, P)
    assert ei.value.code == ErrorCode.CONTINUOUS_MOTION_UNSUPPORTED
    assert describe_region(BAND) in ei.value.message


def test_warning_speed_uses_refined_box_and_axis_words() -> None:
    box = Rect(90.0, 300.0, 744.0, 196.0)
    one = RegimeReport([AMB], 16.0, GRID, (0.0, 0.6))
    d = decide_mode(one, [_scroller(v=37.6, axis="y", box=box)], _scan(), FRAME, P)
    msg = d.warnings[0].message
    assert describe_region(box) in msg and "moving down at about 38 px/s" in msg
    up = replace(_scroller(v=-250.0, axis="y"))
    assert (
        "moving up at about 250 px/s"
        in decide_mode(one, [up], _scan(), FRAME, P).warnings[0].message
    )


# --------------------------------------------------------------------------------------------
# Acceptance run: one scroller row split into activity pieces (flat placeholder cards)
# --------------------------------------------------------------------------------------------


def _piece(x: float, y: float, w: float, h: float, *, axis: str = "x", v: float | None = 35.0,
           scroller: bool = True) -> ScrollerAnalysis:  # fmt: skip
    return ScrollerAnalysis(region=AmbientRegion(Rect(x, y, w, h), 30, 0.8), is_scroller=scroller,
                            axis=axis if scroller else None, autoplay_velocity=v)  # fmt: skip


def _report(pieces: list[ScrollerAnalysis]) -> RegimeReport:
    return RegimeReport([p.region for p in pieces], 16.0, (50, 57), (0.0, 0.6))


def test_pieces_of_one_row_merge_across_the_frame():
    """The run-1 replica: three pieces of one 744 px row (only card edges / titles changed in
    the lead window), local speeds differing with the card scaling (40 vs 34 px/s)."""
    pieces = [
        _piece(729.5, 298.7, 111.0, 188.6, v=40.1),
        _piece(190.3, 298.7, 111.0, 188.6, v=34.6),
        _piece(539.2, 314.4, 111.0, 157.2, v=34.1),
    ]
    merged = merge_row_pieces(_report(pieces), pieces, (904.0, 786.0))
    assert merged is not None and len(merged.ambient) == 1
    a = merged.ambient[0]
    assert a.merged and a.cells == 90
    assert (a.rect_css.x, a.rect_css.w) == (0.0, 904.0)  # the whole frame along the axis
    assert a.rect_css.y == pytest.approx(298.7) and a.rect_css.y2 == pytest.approx(487.3)


@pytest.mark.parametrize(
    "pieces",
    [
        # stacked rows (no overlap across the axis)
        [_piece(100, 100, 300, 80), _piece(100, 300, 300, 80)],
        # opposite directions
        [_piece(100, 100, 200, 80, v=35.0), _piece(400, 100, 200, 80, v=-35.0)],
        # very different speeds
        [_piece(100, 100, 200, 80, v=20.0), _piece(400, 100, 200, 80, v=60.0)],
        # vertical tickers side by side: aligned across x, but their axis is y
        [_piece(100, 100, 80, 400, axis="y"), _piece(300, 100, 80, 400, axis="y")],
        # one of them is not a scroller
        [_piece(100, 100, 200, 80), _piece(400, 100, 200, 80, scroller=False)],
    ],
)
def test_unrelated_regions_do_not_merge(pieces):
    assert merge_row_pieces(_report(pieces), pieces, (1280.0, 800.0)) is None
