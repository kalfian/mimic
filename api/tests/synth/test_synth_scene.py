"""Compositor accuracy: rendered pixels must agree with the scene's own geometry (no codec)."""

from __future__ import annotations

import numpy as np
import pytest

from .animate import Timeline, Tween
from .scene import Node, Renderer, Scene

BG = (200.0, 200.0, 200.0)


def _scene(nodes: list[Node], tweens: list[Tween] | None = None, w: int = 200, h: int = 160):
    return Scene(
        width=w, height=h, background=BG, static_nodes=[], nodes=nodes,
        timeline=Timeline(list(tweens or [])), cursor=None, duration_s=1.0,
    )  # fmt: skip


def _ink(frame: np.ndarray) -> np.ndarray:
    """Coverage of a black shape over the BG gray (0..1 per pixel)."""
    return (BG[0] - frame[..., 0].astype(np.float64)) / BG[0]


def _centroid(cov: np.ndarray) -> tuple[float, float]:
    yy, xx = np.mgrid[0 : cov.shape[0], 0 : cov.shape[1]]
    m = cov.sum()
    # +0.5: pixel i covers [i, i+1) in CSS/edge coordinates
    return (float((xx * cov).sum() / m) + 0.5, float((yy * cov).sum() / m) + 0.5)


@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("tx,ty", [(0.0, 0.0), (0.25, -0.5), (3.7, 1.3)])
def test_subpixel_translate(ratio: int, tx: float, ty: float) -> None:
    box = Node(id="b", x=60, y=40, w=60, h=50, fill=(0, 0, 0), tx=tx, ty=ty)
    r = Renderer(_scene([box]), ratio)
    cx, cy = _centroid(_ink(r.frame(0.0)))
    assert cx / ratio == pytest.approx(90 + tx, abs=0.02)
    assert cy / ratio == pytest.approx(65 + ty, abs=0.02)
    assert _ink(r.frame(0.0)).sum() / ratio**2 == pytest.approx(60 * 50, rel=2e-3)


def test_scale_about_origin() -> None:
    center = Node(id="c", x=60, y=40, w=60, h=50, fill=(0, 0, 0), scale=1.06)
    top = Node(id="t", x=60, y=40, w=60, h=50, fill=(0, 0, 0), scale=1.06, origin=(0.5, 0.0))
    for node, want_cy in ((center, 65.0), (top, 40 + 25 * 1.06)):
        cov = _ink(Renderer(_scene([node])).frame(0.0))
        cx, cy = _centroid(cov)
        assert cx == pytest.approx(90.0, abs=0.02)
        assert cy == pytest.approx(want_cy, abs=0.02)
        assert cov.sum() == pytest.approx(60 * 50 * 1.06**2, rel=2e-3)


def test_child_transform_is_relative_to_parent() -> None:
    child = Node(id="k", x=10, y=10, w=20, h=20, fill=(0, 0, 0))
    parent = Node(id="p", shape="group", x=50, y=40, w=100, h=80, children=[child], ty=-8.0)
    sc = _scene([parent])
    assert sc.box("k", 0.0) == pytest.approx((60.0, 42.0, 20.0, 20.0))
    cx, cy = _centroid(_ink(Renderer(sc).frame(0.0)))
    assert (cx, cy) == (pytest.approx(70.0, abs=0.02), pytest.approx(52.0, abs=0.02))


def test_flat_color_and_color_tween() -> None:
    node = Node(id="b", x=40, y=40, w=80, h=60, fill=(59, 130, 246))
    tw = Tween("b", "background-color", "fwd", 0.1, 0.2, (59, 130, 246), (29, 78, 216), "linear")
    r = Renderer(_scene([node], [tw]))
    assert tuple(r.frame(0.0)[70, 80]) == (246, 130, 59)  # BGR
    assert tuple(r.frame(0.2)[70, 80]) == (231, 104, 44)  # halfway, sRGB lerp
    assert tuple(r.frame(0.5)[70, 80]) == (216, 78, 29)


def test_group_opacity_is_linear() -> None:
    """Group opacity: a child over its parent fades as one layer (no double blending)."""
    child = Node(id="k", x=10, y=10, w=20, h=20, fill=(0, 0, 0))
    parent = Node(id="p", x=40, y=40, w=80, h=60, fill=(255, 255, 255), opacity=0.5,
                  children=[child])  # fmt: skip
    f = Renderer(_scene([parent])).frame(0.0)
    assert int(f[60, 60, 0]) == round(200 * 0.5 + 0 * 0.5)  # child pixel
    assert int(f[90, 100, 0]) == round(200 * 0.5 + 255 * 0.5)  # parent pixel


def test_clip_height_and_overflow_hidden() -> None:
    panel = Node(id="p", x=40, y=20, w=80, h=100, fill=(0, 0, 0), clip_h=30.5)
    cov = _ink(Renderer(_scene([panel])).frame(0.0))
    assert cov.sum() / 80 == pytest.approx(30.5, abs=0.02)

    inner = Node(id="i", x=0, y=50, w=80, h=50, fill=(0, 0, 0))
    clipper = Node(id="p", x=40, y=20, w=80, h=100, fill=(255, 255, 255), clip_children=True,
                   clip_h=60.0, children=[inner])  # fmt: skip
    sc = _scene([clipper])
    cov = _ink(Renderer(sc).frame(0.0))
    assert cov[:, 40:120].clip(0).sum() / 80 == pytest.approx(10.0, abs=0.02)  # rows 70..80
    assert cov[81:, :].max() <= 0.0
    assert sc.box("p", 0.0) == pytest.approx((40.0, 20.0, 80.0, 60.0))
    assert sc.box("i", 0.0) == pytest.approx((40.0, 70.0, 80.0, 10.0))


def test_frame_dedup_returns_identical_frames_when_static() -> None:
    node = Node(id="b", x=40, y=40, w=80, h=60, fill=(0, 0, 0))
    tw = Tween("b", "translateX", "fwd", 0.5, 0.2, 0.0, 10.0, "ease-out")
    r = Renderer(_scene([node], [tw]))
    a, b = r.frame(0.1), r.frame(0.2)
    assert a is b
    assert not np.array_equal(a, r.frame(0.6))


# ---- drivers (PLAN-continuous §8.1) -----------------------------------------------------------


def test_driver_overrides_translate_and_enters_the_dedup_key() -> None:
    child = Node(id="k", x=10, y=10, w=20, h=20, fill=(0, 0, 0))
    track = Node(id="tr", shape="group", x=40, y=40, w=100, h=60, children=[child], tx=99.0)
    sc = _scene([track])
    sc = Scene(
        width=sc.width, height=sc.height, background=BG, static_nodes=[], nodes=[track],
        timeline=Timeline(), cursor=None, duration_s=1.0,
        drivers={"tr": lambda t: (10.0 * t, -2.0)},
    )  # fmt: skip
    assert sc.box("k", 0.5) == pytest.approx((55.0, 48.0, 20.0, 20.0))  # base tx ignored
    r = Renderer(sc)
    cx, cy = _centroid(_ink(r.frame(0.25)))
    assert (cx, cy) == (pytest.approx(62.5, abs=0.02), pytest.approx(58.0, abs=0.02))
    assert r.frame(0.25) is not r.frame(0.5)  # driver output is part of the state key
    assert len(sc.state_key(0.1)) == 3


def test_scene_without_drivers_keeps_its_key_shape() -> None:
    node = Node(id="b", x=40, y=40, w=80, h=60, fill=(0, 0, 0))
    sc = _scene([node], [Tween("b", "translateX", "fwd", 0.5, 0.2, 0.0, 10.0, "ease-out")])
    assert sc.drivers == {}
    key = sc.state_key(0.6)
    assert len(key) == 2 and key[1] is None  # (tween values, cursor) as before drivers existed


def test_driver_validation() -> None:
    node = Node(id="b", x=40, y=40, w=80, h=60, fill=(0, 0, 0))
    with pytest.raises(ValueError, match="unknown node"):
        Scene(200, 160, BG, [], [node], Timeline(), None, 1.0, drivers={"x": lambda t: (0, 0)})
    tw = Tween("b", "translateX", "fwd", 0.1, 0.2, 0.0, 5.0, "linear")
    with pytest.raises(ValueError, match="driven"):
        Scene(200, 160, BG, [], [node], Timeline([tw]), None, 1.0, drivers={"b": lambda t: (0, 0)})
    static = Node(id="s", x=0, y=0, w=10, h=10)
    with pytest.raises(ValueError, match="static node"):
        Scene(200, 160, BG, [static], [], Timeline(), None, 1.0, drivers={"s": lambda t: (0, 0)})
