"""Alignment + per-frame tracking on numpy-rendered images with known transforms (Track M1)."""

from __future__ import annotations

import numpy as np
import pytest

from app.models.measure import Rect
from app.pipeline.params import DEFAULT_PARAMS as P
from app.pipeline.track import (
    IDENTITY,
    Alignment,
    SegmentSpan,
    affine_preferred,
    box_from_union,
    compose,
    decompose,
    estimate_transform,
    geometric_series,
    invert,
    map_box,
    snap_origin,
    track_element,
    transform_origin,
    warp_css_to_px,
    warp_px_to_css,
)
from tests.unit.m1_helpers import (
    base_canvas,
    card_sprite,
    composite,
    scale_about,
    translate,
    window_frames,
)

CARD = card_sprite(180, 200)


def scene(m: np.ndarray) -> np.ndarray:
    img = base_canvas(480, 320)
    composite(img, CARD, m)
    return img


A = scene(translate(140, 70))
BOX = (124, 54, 212, 232)  # card + 16 px


@pytest.mark.parametrize("dx,dy", [(0.0, -8.25), (5.3, 0.0), (-2.5, 3.75)])
def test_estimate_translation(dx: float, dy: float) -> None:
    al = estimate_transform(A, scene(translate(140 + dx, 70 + dy)), BOX, P)
    assert al.warp[0, 2] == pytest.approx(dx, abs=0.15)
    assert al.warp[1, 2] == pytest.approx(dy, abs=0.15)
    assert decompose(al.warp).scale == pytest.approx(1.0, abs=0.003)
    assert al.rho > 0.98


def test_estimate_scale_about_center() -> None:
    m = scale_about(1.06, 230, 170) @ np.vstack([translate(140, 70), [0, 0, 1]])
    al = estimate_transform(A, scene(m), BOX, P)
    assert decompose(al.warp).scale == pytest.approx(1.06, abs=0.004)
    assert transform_origin(al.warp, Rect(140, 70, 180, 200), P) == "center"


def test_affine_needs_to_earn_its_dofs() -> None:
    aniso = Alignment(np.array([[1.0, 0, 0], [0, 1.05, 0]]), 0.99, "ecc")
    assert not affine_preferred(0.985, aniso)  # small gain, anisotropic → translation wins
    iso = Alignment(np.array([[1.04, 0, 0], [0, 1.04, 0]]), 0.999, "ecc")
    assert affine_preferred(0.95, iso)
    sheared = Alignment(np.array([[1.0, 0.05, 0], [0, 1.0, 0]]), 0.9999, "ecc")
    assert not affine_preferred(0.9, sheared)


def test_box_from_union_inverts_translation_and_scale() -> None:
    a = Rect(100, 50, 200, 100)
    for m in (np.array([[1.0, 0, 0], [0, 1.0, -8]]), scale_about(1.1, 200, 100)[:2]):
        u = a.union(map_box(m, a))
        r = box_from_union(u, m)
        assert (r.x, r.y, r.w, r.h) == pytest.approx((a.x, a.y, a.w, a.h), abs=1e-6)


def test_warp_css_px_roundtrip_and_compose() -> None:
    m = np.array([[1.02, 0, 3.0], [0, 1.02, -4.0]])
    px = warp_css_to_px(m, 2.0, (10.0, 20.0))
    assert np.allclose(warp_px_to_css(px, 2.0, (10.0, 20.0)), m)
    assert np.allclose(compose(m, invert(m)), IDENTITY)


def test_snap_origin() -> None:
    assert snap_origin(0.5, 0.02, 0.2) == "top center"
    assert snap_origin(0.0, 1.0, 0.2) == "bottom left"
    assert snap_origin(0.35, 0.5, 0.1) == "35% 50%"


def test_track_element_recovers_per_frame_translation() -> None:
    fps = 60.0
    dys = [-8.0 * min(max((i - 10) / 15, 0.0), 1.0) for i in range(40)]
    frames = [scene(translate(140, 70 + dy)) for dy in dys]
    wf = window_frames(frames, fps)
    box = Rect(140, 70, 180, 200)
    ft = track_element(wf, frames[0], box, np.array([[1.0, 0, 0], [0, 1.0, -8.0]]), P)
    assert ft.visible.all()
    meas = ft.warps[:, 1, 2]
    assert np.max(np.abs(meas - np.array(dys))) < 0.15
    span = SegmentSpan("fwd", 10 / fps, 25 / fps, 0.0, wf.times[-1])
    series, hints = geometric_series("e1", ft.warps, wf.times, ft.visible, ft.rho, box, span, P)
    ty = next(s for s in series if s.property == "translateY")
    assert ty.v_start == pytest.approx(0.0, abs=0.1) and ty.v_end == pytest.approx(-8.0, abs=0.15)
    assert ty.from_value.number == pytest.approx(0.0, abs=0.1)
    assert hints == []


def test_track_element_reuses_duplicate_frames() -> None:
    """Phase 5 perf: duplicate frames copy their run's result instead of re-running ECC."""
    fps = 60.0
    dys = [-8.0 * min(max((i - 10) / 15, 0.0), 1.0) for i in range(40)]
    frames = [scene(translate(140, 70 + dy)) for dy in dys]
    wf = window_frames(frames, fps)
    for i in range(1, 40):
        wf.is_dup[i] = dys[i] == dys[i - 1]
    fwd = (frames[0], Rect(140, 70, 180, 200), np.array([[1.0, 0, 0], [0, 1.0, -8.0]]))
    bwd = (frames[-1], Rect(140, 62, 180, 200), np.array([[1.0, 0, 0], [0, 1.0, 8.0]]))
    for backwards, (src, box, final) in ((False, fwd), (True, bwd)):
        ft = track_element(wf, src, box, final, P, backwards=backwards)
        assert ft.visible.all()
        for i in np.nonzero(wf.is_dup)[0]:
            run_start = max(j for j in range(i + 1) if not wf.is_dup[j])
            assert np.array_equal(ft.warps[i], ft.warps[run_start])
        if not backwards:
            assert np.max(np.abs(ft.warps[:, 1, 2] - np.array(dys))) < 0.15
