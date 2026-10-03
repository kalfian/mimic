"""Blend coefficient, colours, opacity (Track M1)."""

from __future__ import annotations

import numpy as np
import pytest

from app.models.ir import PxValue, RatioValue
from app.models.measure import ElementCandidate, PropertySeries, Rect
from app.pipeline.params import DEFAULT_PARAMS as P
from app.pipeline.photometric import (
    alpha_series,
    bgr_to_lab,
    blend_alpha,
    delta_e76,
    element_colors,
    hex_from_lab,
    measure_photometric,
    opacity_kappa,
    refine_appear_endpoints,
)
from app.pipeline.track import SegmentSpan
from tests.unit.m1_helpers import (
    base_canvas,
    composite,
    solid_rect_sprite,
    translate,
    window_frames,
)


def test_lab_roundtrip_hex() -> None:
    lab = bgr_to_lab(np.array([246, 130, 59], np.uint8))  # #3B82F6
    assert hex_from_lab(lab) == "#3B82F6"
    assert delta_e76(lab, lab) == 0.0


def test_blend_alpha_linear_mix() -> None:
    rng = np.random.default_rng(0)
    a = rng.integers(0, 120, (30, 30, 3)).astype(np.uint8)
    b = rng.integers(130, 255, (30, 30, 3)).astype(np.uint8)
    mask = np.ones((30, 30), bool)
    for alpha in (0.0, 0.25, 0.6, 1.0):
        f = (a * (1 - alpha) + b * alpha).round().astype(np.uint8)
        al, q = blend_alpha(f, a, b, mask)
        assert al == pytest.approx(alpha, abs=0.01) and q > 0.98


def _button(color) -> np.ndarray:
    img = base_canvas(320, 200)
    composite(img, solid_rect_sprite(140, 40, color, 6), translate(90, 80))
    return img


def test_fill_colors_and_kappa_hue_check() -> None:
    a, b = _button((246, 130, 59)), _button((216, 78, 29))
    mask = np.zeros(a.shape[:2], bool)
    mask[82:118, 92:228] = True
    pair = element_colors(a, b, mask, mask, text=False, k=1.0, params=P)
    assert delta_e76(pair.lab_a, bgr_to_lab(np.array([246, 130, 59], np.uint8))) < 2
    assert delta_e76(pair.lab_b, bgr_to_lab(np.array([216, 78, 29], np.uint8))) < 2
    # a darker blue is not a "dimmed" blue: the hue check rejects the opacity model
    kap = opacity_kappa(a, b, mask, 1.0)
    assert kap is not None and kap[1] <= P.photometric.opacity_kappa_r2


def test_dimmed_element_is_opacity() -> None:
    bg = np.array(base_canvas(320, 200)[150, 150], np.float64)
    full = np.array([60.0, 60.0, 220.0])
    dim = bg + 0.5 * (full - bg)
    a, b = _button(tuple(int(v) for v in dim)), _button((60, 60, 220))
    mask = np.zeros(a.shape[:2], bool)
    mask[80:120, 90:230] = True  # the whole element: the ring around it is the background
    kappa, r2 = opacity_kappa(a, b, mask, 1.0)
    assert kappa == pytest.approx(0.5, abs=0.03) and r2 > 0.9


def test_measure_photometric_color_series() -> None:
    cols = [(246, 130, 59)] * 10 + [(231, 104, 44)] + [(216, 78, 29)] * 10
    frames = [_button(c) for c in cols]
    wf = window_frames(frames, 60.0)
    mask = np.zeros(frames[0].shape[:2], bool)
    mask[82:118, 92:228] = True
    el = ElementCandidate("e1", Rect(90, 80, 140, 40), Rect(90, 80, 140, 40), None,
                          "photometric", 1.0)  # fmt: skip
    span = SegmentSpan("fwd", 9 / 60, 11 / 60, 0.0, wf.times[-1])
    res = measure_photometric([el], {"e1": mask}, {"e1": (Rect(90, 80, 140, 40),) * 2},
                              {"e1": np.eye(2, 3)}, {}, frames[0], frames[-1],
                              np.full(mask.shape, 255, np.uint8), [(wf, [span])], P)  # fmt: skip
    (s,) = res.series
    assert s.property == "background-color" and s.unit == "color"
    assert s.v_start == pytest.approx(0, abs=0.02) and s.v_end == pytest.approx(1, abs=0.02)
    assert s.values[10] == pytest.approx(0.5, abs=0.08)
    assert s.from_value.color == res.colors["e1"].hex_a


def test_alpha_series_appear_static() -> None:
    a = base_canvas(200, 120)
    spr = solid_rect_sprite(80, 40, (30, 30, 30))
    frames = []
    for op in np.linspace(0, 1, 11):
        f = a.copy()
        composite(f, spr, translate(60, 40), opacity=float(op))
        frames.append(f)
    wf = window_frames(frames, 60.0)
    mask = np.zeros(a.shape[:2], bool)
    mask[42:78, 62:138] = True
    al, q = alpha_series(wf, frames[0], frames[-1], mask)
    assert np.allclose(al, np.linspace(0, 1, 11), atol=0.03)


def _ser(prop, unit, times, vals, v0, v1) -> PropertySeries:
    cls = RatioValue if unit == "ratio" else PxValue
    return PropertySeries("e1", prop, "fwd", times, vals, np.ones_like(vals), v0, v1, unit,
                          cls(number=v0), cls(number=v1))  # fmt: skip


def test_refine_appear_endpoints_extrapolates_to_zero_opacity() -> None:
    t = np.arange(10) / 60
    op = np.linspace(0.0, 1.0, 10)
    ty = 8.0 * (1 - op)  # shares timing with the fade
    vis = op > 0.3  # only tracked once visible
    s_op = _ser("opacity", "ratio", t, op, 0.0, 1.0)
    s_ty = _ser("translateY", "px", t[vis], ty[vis], float(ty[vis][0]), 0.0)
    refine_appear_endpoints([s_op, s_ty])
    assert s_ty.v_start == pytest.approx(8.0, abs=0.05)
    assert s_ty.from_value.number == pytest.approx(8.0, abs=0.05)


def test_alpha_series_cache_matches_per_frame_blend() -> None:
    """Phase 5 perf: the shared A/B precomputation gives exactly blend_alpha's numbers."""
    a = base_canvas(200, 120)
    spr = solid_rect_sprite(80, 40, (30, 30, 30))
    frames = []
    for op in np.linspace(0, 1, 9):
        f = a.copy()
        composite(f, spr, translate(60, 40), opacity=float(op))
        frames.append(f)
    wf = window_frames(frames, 60.0)
    cursor = np.zeros(a.shape[:2], np.uint8)
    cursor[50:60, 70:80] = 255
    wf.cursor_masks[4] = cursor
    mask = np.zeros(a.shape[:2], bool)
    mask[42:78, 62:138] = True
    warps = np.repeat(np.array([[[1.0, 0, 0], [0, 1.0, 0]]]), 9, axis=0)
    warps[5:, 0, 2] = 1.5  # B moves by 1.5 px from frame 5 on
    al, q = alpha_series(wf, frames[0], frames[-1], mask, warps_b=warps)
    for i, f in enumerate(frames):
        from app.pipeline.photometric import _warp

        bt = _warp(frames[-1], warps[i], a.shape)
        mt = _warp(mask.astype(np.uint8), warps[i], a.shape) > 0
        ref = blend_alpha(f, frames[0], bt, mt, cursor=wf.cursor_masks[i])
        assert (al[i], q[i]) == pytest.approx(ref, nan_ok=True, abs=0, rel=0)
