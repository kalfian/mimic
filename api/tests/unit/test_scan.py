"""A2 scan (Track M1): hysteresis segmentation, pairing, global motion, cursor separation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.core.errors import ErrorCode, PipelineError
from app.pipeline.params import DEFAULT_PARAMS as P
from app.pipeline.scan import (
    Pass1Accumulator,
    build_scan_result,
    energy_thresholds,
    scan_video,
    segment_energy,
)
from tests.unit.m1_helpers import (
    HoverScene,
    base_canvas,
    composite,
    render_video,
    requires_ffmpeg,
    scale1,
    solid_rect_sprite,
    textured_patch,
    translate,
)

FPS = 30.0


def _times(n: int) -> np.ndarray:
    return np.arange(n) / FPS


# ---- segment_energy (pure) --------------------------------------------------------------------


def test_thresholds_floor_and_mins() -> None:
    th = energy_thresholds(np.zeros(50), P)
    assert th.hi == P.scan.energy_min_hi_px2
    assert th.lo == P.scan.energy_min_lo_px2


def test_single_run_with_soft_onset_and_settle() -> None:
    e = np.zeros(60)
    e[20] = 20.0  # soft onset (between lo and hi) is included
    e[21:30] = 500.0
    runs = segment_energy(e, _times(60), P)
    assert len(runs) == 1
    assert (runs[0].i0, runs[0].i1) == (20, 29)
    assert runs[0].settled


def test_runs_closer_than_merge_gap_are_merged() -> None:
    e = np.zeros(80)
    e[10:15] = 500.0
    e[18:22] = 500.0  # 3 quiet frames = 100 ms < 150 ms
    e[40:45] = 500.0  # far away → separate
    runs = segment_energy(e, _times(80), P)
    assert [(r.i0, r.i1) for r in runs] == [(10, 21), (40, 44)]


def test_compression_blip_dropped() -> None:
    e = np.zeros(40)
    e[10] = 100.0  # ΣE < energy_min_total_px2
    assert segment_energy(e, _times(40), P) == []


def test_run_reaching_end_is_not_settled() -> None:
    e = np.zeros(30)
    e[25:] = 500.0
    runs = segment_energy(e, _times(30), P)
    assert len(runs) == 1 and not runs[0].settled


# ---- accumulator + result (no codec) ----------------------------------------------------------


def _accumulate(frames: list[np.ndarray]):
    acc = Pass1Accumulator(params=P, scale=scale1(), image_scale=1.0,
                           frame_css=(frames[0].shape[1], frames[0].shape[0]))  # fmt: skip
    for i, f in enumerate(frames):
        g = f if f.ndim == 2 else f.mean(axis=2).astype(np.uint8)
        acc.add(i / FPS, g)
    return acc.finish()


def _card_frames(n: int, dy_of_t) -> list[np.ndarray]:
    card = solid_rect_sprite(120, 90, (90, 90, 200))
    out = []
    for i in range(n):
        img = base_canvas(320, 240)
        composite(img, card, translate(100, 70 + dy_of_t(i / FPS)))
        out.append(img)
    return out


def test_static_video_fails_no_motion() -> None:
    stats = _accumulate(_card_frames(30, lambda t: 0.0))
    with pytest.raises(PipelineError) as ei:
        build_scan_result(stats, scale1(), P)
    assert ei.value.code == ErrorCode.NO_MOTION_DETECTED


def test_motion_at_start_fails_no_stable_state() -> None:
    stats = _accumulate(_card_frames(40, lambda t: -min(t, 0.3) * 30))
    with pytest.raises(PipelineError) as ei:
        build_scan_result(stats, scale1(), P)
    assert ei.value.code == ErrorCode.NO_STABLE_STATE


def test_round_trip_detected_for_press() -> None:
    def dy(t: float) -> float:  # down 6 px and back within 0.3 s
        u = (t - 0.6) / 0.3
        return 0.0 if not 0 <= u <= 1 else 6.0 * np.sin(np.pi * u)

    res = build_scan_result(_accumulate(_card_frames(45, dy)), scale1(), P)
    assert len(res.segments) == 1
    assert res.primary.round_trip and res.primary.reverse is None


def test_global_scroll_is_unsupported() -> None:
    page = textured_patch(320, 1000, seed=3)
    frames = []
    for i in range(45):
        off = int(min(max((i / FPS - 0.5) / 0.4, 0.0), 1.0) * 200)
        frames.append(page[off : off + 240].copy())
    with pytest.raises(PipelineError) as ei:
        build_scan_result(_accumulate(frames), scale1(), P)
    assert ei.value.code == ErrorCode.UNSUPPORTED_MOTION


# ---- end to end on an encoded clip ------------------------------------------------------------


@pytest.fixture(scope="module")
def hover_clip(tmp_path_factory: pytest.TempPathFactory):
    scene = HoverScene()
    path = tmp_path_factory.mktemp("scan") / "hover.mp4"
    probe, _ = render_video(scene, path, 3.2)
    return scene, path, probe


@requires_ffmpeg
def test_scan_video_hover(hover_clip) -> None:
    scene, path, probe = hover_clip
    res, stats = scan_video(Path(path), probe, scale1(), P)
    assert len(res.segments) == 2
    fwd, rev = res.primary.forward, res.primary.reverse
    assert rev is not None and not res.primary.round_trip
    assert abs(fwd.start_s - scene.fwd_t0) <= 0.1
    assert abs(rev.start_s - scene.rev_t0) <= 0.1
    # the ROI covers the card (cursor excluded)
    cx, cy = scene.card_xy
    assert fwd.roi_css.x <= cx + 2 and fwd.roi_css.x2 >= cx + scene.card_wh[0] - 2
    assert res.cursor.visible
    assert stats.thumbs.shape[0] == len(res.frame_times)
    # pointer energy is not UI energy: nothing while only the cursor moves (0.4–1.1 s)
    quiet = (res.frame_times > 0.45) & (res.frame_times < 1.1)
    assert float(res.energy[quiet].max()) == 0.0
