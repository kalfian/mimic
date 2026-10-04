"""Scroller displacement: shift estimation, coherence test, tracker, region reader
(PLAN-continuous §4.1–§4.3). Textures are rendered in-test (numpy + sub-pixel warpAffine)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.models.measure import AmbientRegion, Rect
from app.pipeline import decode
from app.pipeline.continuous import displacement as disp
from app.pipeline.continuous.measure import analyse_stream, region_crop
from app.pipeline.params import DEFAULT_PARAMS
from tests.unit.m1_helpers import encode, make_probe, requires_ffmpeg, scale1


def texture(w: int, h: int, *, sigma: float = 2.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = cv2.GaussianBlur(rng.random((h, w)).astype(np.float32), (0, 0), sigma)
    return ((t - t.min()) / (t.max() - t.min()) * 200 + 20).astype(np.float32)


BIG = texture(2000, 200)


def view(x: float, big: np.ndarray = BIG, w: int = 600, h: int = 160, x0: int = 600) -> np.ndarray:
    """Crop of ``big`` whose content is shifted right by ``x`` px (sub-pixel, cubic)."""
    m = np.float32([[1, 0, x], [0, 1, 0]])
    sh = cv2.warpAffine(big, m, (big.shape[1], big.shape[0]), flags=cv2.INTER_CUBIC)
    return np.ascontiguousarray(sh[20 : 20 + h, x0 : x0 + w])


# --------------------------------------------------------------------------------------------
# estimate_shift
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("d", [0.13, 0.37, 0.5, 0.71, 1.25, 3.3, -0.42, -2.6])
def test_subpixel_shift_within_0_1px(d: float) -> None:
    a, b = view(0.0), view(d)
    for d_hat in (None, d * 0.9, 0.0):
        est = disp.estimate_shift(a, b, d_hat)
        assert est.d == pytest.approx(d, abs=0.1), (d_hat, est)
        assert est.q > 0.95


@pytest.mark.parametrize("d", [7.9, 15.4, 33.3, 61.7, 90.2, 119.6, -45.5, -118.3])
def test_large_shift_within_0_5px(d: float) -> None:
    a, b = view(0.0), view(d)
    est = disp.estimate_shift(a, b, d * 0.9)  # constant-velocity prediction ±10 %
    assert est.d == pytest.approx(d, abs=0.5)
    if abs(d) < 60:  # without a prediction phase correlation alone resolves moderate shifts
        assert disp.estimate_shift(a, b, None).d == pytest.approx(d, abs=0.5)


@pytest.mark.parametrize("d", [20.3, 60.6, 110.2])
def test_blurred_frames_within_0_5px(d: float) -> None:
    """Recorder frame blending / compression smear: either frame box-blurred along the axis."""
    length = int(d / 2)
    blurred_cur = disp.axis_box_blur(view(d), length)
    est = disp.estimate_shift(view(0.0), blurred_cur, d * 0.95)
    assert est.d == pytest.approx(d, abs=0.5)
    blurred_ref = disp.axis_box_blur(view(0.0), length)
    est = disp.estimate_shift(blurred_ref, view(d), d * 0.95)
    assert est.d == pytest.approx(d, abs=0.5)


def test_axis_box_blur_is_centred() -> None:
    img = np.zeros((4, 41), np.float32)
    img[:, 20] = 1.0
    for length in (4, 5, 10):  # even lengths would shift the centroid by ½ px
        out = disp.axis_box_blur(img, length)
        centroid = float((out[0] * np.arange(41)).sum() / out[0].sum())
        assert centroid == pytest.approx(20.0, abs=1e-6)


def test_periodic_texture_prediction_resolves_wrap_ambiguity() -> None:
    """Identical cards every 100 px: a 70 px shift aliases to −30 without a prediction."""
    tile = texture(100, 200, seed=3)
    big = np.concatenate([tile] * 20, axis=1)
    a, b = view(0.0, big), view(70.0, big)
    est = disp.estimate_shift(a, b, 66.0)
    assert est.d == pytest.approx(70.0, abs=0.5)
    est = disp.estimate_shift(a, b, -27.0)  # a prediction near the alias picks the alias
    assert est.d == pytest.approx(-30.0, abs=0.5)


def test_phase_shift_does_not_mutate_inputs() -> None:
    """OpenCV 5.0 ``phaseCorrelate(a, b, window)`` windows its inputs in place; the wrapper
    must not."""
    a, b = view(0.0), view(3.0)
    a0, b0 = a.copy(), b.copy()
    dx, _, _ = disp.phase_shift(a, b)
    assert dx == pytest.approx(3.0, abs=0.5)  # 5×5-centroid estimate (initial value only)
    assert np.array_equal(a, a0) and np.array_equal(b, b0)


# --------------------------------------------------------------------------------------------
# coherence / axis test, strip split, region refinement
# --------------------------------------------------------------------------------------------


def _frames(
    dx: float, dy: float, n: int = 30, seed: int = 0
) -> tuple[list[np.ndarray], list[float]]:
    big = texture(900, 600, seed=seed)
    frames = []
    for i in range(n):
        m = np.float32([[1, 0, dx * i], [0, 1, dy * i]])
        sh = cv2.warpAffine(big, m, (900, 600), flags=cv2.INTER_LINEAR)
        frames.append(np.ascontiguousarray(sh[150:350, 150:650]))
    return frames, [i / 60 for i in range(n)]


def test_coherence_horizontal_scroller() -> None:
    frames, times = _frames(0.7, 0.0)
    res = disp.coherence_test(frames, times, 1.0)
    assert res.is_scroller and res.axis == "x"
    assert res.velocity == pytest.approx(42.0, rel=0.05)  # seeds the first prediction
    assert res.median_perp_px < 0.25


def test_coherence_vertical_scroller() -> None:
    frames, times = _frames(0.0, -1.5)
    res = disp.coherence_test(frames, times, 1.0)
    assert res.is_scroller and res.axis == "y" and res.velocity < 0


def test_coherence_rejects_two_axis_motion() -> None:
    frames, times = _frames(1.0, 0.8)
    res = disp.coherence_test(frames, times, 1.0)
    assert not res.is_scroller and res.reason == "two_axis"


def test_coherence_rejects_untrackable_noise() -> None:
    rng = np.random.default_rng(0)
    frames = [rng.random((120, 300)).astype(np.float32) * 255 for _ in range(20)]
    res = disp.coherence_test(frames, [i / 30 for i in range(20)], 1.0)
    assert not res.is_scroller and res.reason == "untrackable"


def test_strip_split_finds_opposing_rows() -> None:
    top, _ = _frames(1.0, 0.0, seed=1)
    bottom, times = _frames(-1.0, 0.0, seed=2)
    frames = [np.vstack([a[:96], b[:96]]) for a, b in zip(top, bottom, strict=True)]
    vel, _ = disp.strip_velocities(frames, times, "x", 16, 1.0)
    groups = disp.cluster_strips(vel)
    assert len(groups) == 2
    signs = sorted(float(np.sign(np.median(vel[g]))) for g in groups)
    assert signs == [-1.0, 1.0]


def test_refine_box_excludes_static_margin() -> None:
    frames, _ = _frames(1.0, 0.0, n=10)
    padded = []
    for f in frames:
        p = np.full((260, 560), 240.0, np.float32)
        p[30:230, 30:530] = f
        padded.append(p)
    x0, y0, x1, y1 = disp.refine_box(padded)
    assert abs(x0 - 30) <= 3 and abs(y0 - 30) <= 3 and abs(x1 - 530) <= 3 and abs(y1 - 230) <= 3


# --------------------------------------------------------------------------------------------
# tracker
# --------------------------------------------------------------------------------------------


def test_tracker_constant_velocity_with_duplicates() -> None:
    """60 fps, 40 px/s for 1 s then −900 px/s; recorder duplicates are skipped (merged dt)."""
    frames, times, truth = [], [], []
    for i in range(90):
        t = i / 60
        x = 40 * t if t < 1.0 else 40.0 - 900 * (t - 1.0)
        if i in (20, 21, 50):  # duplicate of the previous frame
            x = truth[-1]
        frames.append(view(x, w=500, h=150))
        times.append(t)
        truth.append(x)
    tr = disp.track_frames(frames, times, "x")
    assert len(tr.times) == 90 - 3
    s = tr.series(Rect(0, 0, 500, 150), 0.7)
    want = np.interp(s.times, times, truth)
    assert np.max(np.abs(s.pos - (want - want[0]))) < 0.5
    assert s.quality.min() > 0.9


def test_tracker_has_no_drift_at_fractional_px_per_frame() -> None:
    """0.667 px/frame (40 px/s at 60 fps) on bilinear-rendered frames: summing frame-to-frame
    shifts accumulates the interpolation bias (≈ 2 % speed error); keyframe anchoring keeps the
    speed within 0.5 %."""
    big = texture(2000, 200, sigma=1.2, seed=8)
    frames, times = [], []
    for i in range(120):
        m = np.float32([[1, 0, 40.0 * i / 60], [0, 1, 0]])
        sh = cv2.warpAffine(big, m, (2000, 200), flags=cv2.INTER_LINEAR)
        frames.append(np.ascontiguousarray(sh[20:180, 600:1200]))
        times.append(i / 60)
    s = disp.track_frames(frames, times, "x").series(Rect(0, 0, 600, 160), 0.7)
    speed = np.polyfit(s.times, s.pos, 1)[0]
    assert speed == pytest.approx(40.0, rel=0.005)


def test_stall_mask_drops_lossy_duplicates_inside_motion_only() -> None:
    pos = np.array([0, 1, 2, 3, 3, 3, 6, 7, 8, 8, 8, 8, 8, 8, 9, 10], dtype=float)
    keep = disp.stall_mask(pos)
    # the two stalled frames inside motion go; the long rest (5 frames) stays
    assert keep.tolist() == [True] * 4 + [False, False] + [True] * 10


def test_smoothed_velocity_recovers_slope_on_nonuniform_time() -> None:
    t = np.cumsum(np.r_[0, np.random.default_rng(0).uniform(0.012, 0.022, 60)])
    x = 35.0 * t
    v = disp.smoothed_velocity(t, x)
    np.testing.assert_allclose(v, 35.0, atol=1e-6)


# --------------------------------------------------------------------------------------------
# iter_region + end to end (needs ffmpeg)
# --------------------------------------------------------------------------------------------

W, H = 480, 200
FPS = 30.0


def _strip_frame(pos: float, strip: np.ndarray) -> np.ndarray:
    img = np.full((H, W, 3), 238, np.uint8)
    m = np.float32([[1, 0, pos - 400], [0, 1, 0]])
    band = cv2.warpAffine(strip, m, (W - 80, 120), flags=cv2.INTER_LINEAR)
    img[40:160, 40 : W - 40] = np.clip(band, 0, 255).astype(np.uint8)[..., None]
    return img


@pytest.fixture(scope="module")
def scroller_clip(tmp_path_factory: pytest.TempPathFactory):
    """3 s at 30 fps: 45 px/s right; frames 40 + 41 duplicate frame 39."""
    strip = texture(1400, 120, sigma=2.5, seed=4)
    frames, truth = [], []
    for i in range(int(3 * FPS)):
        x = 45.0 * i / FPS
        if i in (40, 41):
            x = truth[39]
        truth.append(x)
        frames.append(_strip_frame(x, strip))
    # crf 0 (lossless): recorder duplicates are bit-identical; lossy H.264 would add noise
    path = encode(frames, tmp_path_factory.mktemp("cont") / "scroller.mp4", FPS, crf=0)
    return path, make_probe(len(frames), FPS, W, H), np.asarray(truth)


@requires_ffmpeg
def test_iter_region_streams_crop_and_skips_duplicates(scroller_clip) -> None:
    path, probe, _ = scroller_clip
    crop = Rect(32, 32, 416, 136)
    warnings: list = []
    out = list(decode.iter_region(path, probe, scale1(), crop, DEFAULT_PARAMS, warnings=warnings))
    assert warnings == []
    assert len(out) == int(3 * FPS) - 2  # the two duplicates are skipped
    t, f = out[45]
    assert f.dtype == np.float32 and f.shape == (136, 416)
    times = np.array([t for t, _ in out])
    assert np.all(np.diff(times) > 0)
    assert 40 / FPS not in set(np.round(times, 6))  # merged dt instead of zero velocity
    region = AmbientRegion(Rect(40, 40, 400, 120), cells=100, lead_frac=1.0)
    crop_css, crop_px = region_crop(region, probe, scale1(), DEFAULT_PARAMS)
    assert crop_css == Rect(8, 8, 464, 184)
    assert crop_px[2:] == (464, 184)


@requires_ffmpeg
def test_analyse_stream_on_encoded_marquee(scroller_clip) -> None:
    path, probe, truth = scroller_clip
    region = AmbientRegion(Rect(40, 40, 400, 120), cells=100, lead_frac=1.0)
    crop_css, crop_px = region_crop(region, probe, scale1(), DEFAULT_PARAMS)
    frames = decode.iter_region(path, probe, scale1(), crop_css, DEFAULT_PARAMS)
    sa = analyse_stream(frames, region, crop_css, crop_px[2] / crop_css.w)
    assert sa.is_scroller and sa.axis == "x"
    assert sa.autoplay_velocity == pytest.approx(45.0, rel=0.02)
    assert [p.kind for p in sa.phases] == ["autoplay"]
    assert sa.loop_period_px is None
    assert any(w.code == "loop_period_not_observed" for w in sa.warnings)
    assert sa.series is not None
    reg = sa.series.region
    assert abs(reg.x - 40) <= 4 and abs(reg.y - 40) <= 4 and abs(reg.w - 400) <= 8
    want = truth[np.round(sa.series.times * FPS).astype(int)]
    assert np.max(np.abs(sa.series.pos - (want - want[0]))) < 0.5


def test_reported_box_is_unpadded() -> None:
    """The scroller box reported in the IR is the changed pixels themselves (pad=0); the padded
    box (tracking crops) is 2·REFINE_PAD_PX larger, which made every round trip off by 4 px."""
    frames, _ = _frames(1.0, 0.0, n=10)
    padded = []
    for f in frames:
        p = np.full((260, 560), 240.0, np.float32)
        p[30:230, 30:530] = f
        padded.append(p)
    a = disp.refine_box(padded)
    b = disp.refine_box(padded, pad=0)
    assert (a[2] - a[0]) - (b[2] - b[0]) == 2 * disp.REFINE_PAD_PX
    assert abs(b[0] - 30) <= 1 and abs(b[2] - 530) <= 1


def test_changed_span_bridges_static_stretches_along_the_axis() -> None:
    """A row of flat cards changes only where edges / titles pass: the tracking box spans the
    first … last changed column, not just the longest changed stretch."""
    rng = np.random.default_rng(0)
    frames = []
    for i in range(8):
        f = np.full((100, 600), 20.0, np.float32)
        for x0 in (40, 300, 520):  # three textured patches separated by flat stretches
            f[20:80, x0 : x0 + 40] = rng.uniform(0, 255, (60, 40)) if i else 128.0
        frames.append(f)
    box = disp.refine_box(frames)
    assert box[2] - box[0] < 100  # one patch only
    wide = disp.changed_span(frames, box, "x")
    assert wide[0] <= 40 and wide[2] >= 558 and wide[1:4:2] == box[1:4:2]
