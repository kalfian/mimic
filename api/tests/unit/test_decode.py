"""ffmpeg readers + window planning (Track M1)."""

from __future__ import annotations

import numpy as np
import pytest

from app.models.ir import SpecWarning
from app.models.measure import (
    ActiveSegment,
    CursorTrack,
    PrimaryInteraction,
    Rect,
    ScanResult,
)
from app.pipeline.decode import (
    WindowSpec,
    clipped_sides,
    crop_rect_css,
    decode_windows,
    expand_crop,
    iter_pass1,
    pass1_geometry,
    plan_windows,
)
from app.pipeline.params import DEFAULT_PARAMS as P
from tests.unit.m1_helpers import base_canvas, encode, make_probe, requires_ffmpeg, scale1

pytestmark = requires_ffmpeg
FPS = 30.0
N = 45


@pytest.fixture(scope="module")
def clip(tmp_path_factory: pytest.TempPathFactory):
    """1.5 s, 30 fps: a box moves right 2 px/frame between frames 15 and 30, static otherwise."""
    frames = []
    for i in range(N):
        img = base_canvas(320, 240)
        x = 60 + 2 * min(max(i - 15, 0), 15)
        img[100:140, x : x + 50] = (40, 40, 200)
        frames.append(img)
    path = encode(frames, tmp_path_factory.mktemp("dec") / "clip.mp4", FPS)
    return path, make_probe(N, FPS, 320, 240)


def test_pass1_stream(clip) -> None:
    path, probe = clip
    g = pass1_geometry(probe, P)
    assert (g.width, g.height) == (320, 240)  # no upscaling of small sources
    out = list(iter_pass1(path, probe, P))
    assert len(out) == N
    t, f = out[10]
    assert f.shape == (240, 320) and f.dtype == np.uint8
    assert t == pytest.approx(10 / FPS)


def _scan_stub(segs: list[ActiveSegment]) -> ScanResult:
    return ScanResult(
        energy=np.zeros(N), frame_times=np.arange(N) / FPS, segments=segs, stable=[],
        thumbs=np.zeros((N, 2, 2), np.uint8), primary=PrimaryInteraction(forward=segs[0]),
        cursor=CursorTrack(visible=False),
    )  # fmt: skip


def test_plan_windows_clamps_and_warns(clip) -> None:
    _, probe = clip
    seg = ActiveSegment(0.1, 0.6, Rect(60, 100, 80, 40), 1.0)
    later = ActiveSegment(0.7, 0.9, Rect(60, 100, 80, 40), 1.0)
    specs, warns = plan_windows(_scan_stub([seg, later]), probe, P)
    assert len(specs) == 1 and specs[0].segment_id == "fwd"
    assert specs[0].start_s == pytest.approx(0.0)  # clamped to video start
    assert specs[0].end_s == pytest.approx(0.7)  # clamped to the next segment
    assert [w.code for w in warns] == ["short_stable_state"]
    assert all(isinstance(w, SpecWarning) for w in warns)


def test_crop_rect_padding_and_clamp(clip) -> None:
    _, probe = clip
    css, px = crop_rect_css([Rect(60, 100, 80, 40)], probe, scale1(), P)
    assert css.x == pytest.approx(12) and css.y == pytest.approx(52)
    assert px[2] % 2 == 0 and px[3] % 2 == 0
    css2, _ = crop_rect_css([Rect(0, 0, 50, 50)], probe, scale1(), P)
    assert css2.x == 0 and css2.y == 0


def test_decode_windows_crops_times_dups(clip) -> None:
    path, probe = clip
    css, px = crop_rect_css([Rect(60, 100, 80, 40)], probe, scale1(), P)
    seg = ActiveSegment(0.5, 1.0, Rect(60, 100, 80, 40), 1.0)
    wfs, warns = decode_windows(path, probe, scale1(), [WindowSpec("fwd", 0.3, 1.2, seg)], css,
                                px, P)  # fmt: skip
    assert warns == []
    wf = wfs[0]
    assert wf.times[0] == pytest.approx(0.3, abs=1e-6) and wf.times[-1] == pytest.approx(1.2)
    assert wf.crops[0].shape == (px[3], px[2], 3)
    # static frames before the motion are duplicates, moving frames are not
    assert wf.is_dup[1] and not wf.is_dup[np.searchsorted(wf.times, 0.7)]
    assert wf.crop_css == css and wf.cursor_masks[0] is None


def test_frame_count_mismatch_falls_back_to_uniform(clip) -> None:
    path, probe = clip
    bad = make_probe(N + 5, FPS, 320, 240)  # probe claims more frames than decode yields
    css, px = crop_rect_css([Rect(60, 100, 80, 40)], probe, scale1(), P)
    seg = ActiveSegment(0.5, 1.0, Rect(60, 100, 80, 40), 1.0)
    _, warns = decode_windows(path, bad, scale1(), [WindowSpec("fwd", 0.3, 1.2, seg)], css, px, P)
    assert [w.code for w in warns] == ["timestamps_estimated"]


def test_clipped_sides_and_expand(clip) -> None:
    _, probe = clip
    css, _ = crop_rect_css([Rect(100, 100, 40, 40)], probe, scale1(), P)
    m = np.zeros((int(css.h), int(css.w)), bool)
    m[10:20, -1] = True
    assert clipped_sides(m, css, probe, scale1()) == {"right"}
    css2, px2 = expand_crop(css, {"right"}, probe, scale1(), P)
    assert css2.x == pytest.approx(css.x) and css2.x2 > css.x2


def test_long_window_is_subsampled_with_warning(clip) -> None:
    """Phase 5: a window above ``max_window_frames`` is thinned and reported in the IR."""
    import dataclasses

    path, probe = clip
    params = dataclasses.replace(P, decode=dataclasses.replace(P.decode, max_window_frames=10))
    css, px = crop_rect_css([Rect(60, 100, 80, 40)], probe, scale1(), params)
    seg = ActiveSegment(0.5, 1.0, Rect(60, 100, 80, 40), 1.0)
    wfs, warns = decode_windows(path, probe, scale1(), [WindowSpec("fwd", 0.3, 1.2, seg)], css,
                                px, params)  # fmt: skip
    assert len(wfs[0].crops) == 10
    assert [w.code for w in warns] == ["frames_subsampled"]
    w = warns[0]
    assert w.severity == "info"
    assert "28 frames" in w.message and "about 10 fps" in w.message


def test_subsample_warning_names_the_most_reduced_window() -> None:
    from app.pipeline.decode import subsample_warning

    w = subsample_warning([("fwd", 200, 150, 3.3), ("rev", 400, 150, 6.6)])
    assert "400 frames" in w.message and "about 23 fps" in w.message
