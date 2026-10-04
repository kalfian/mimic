"""Appearance truth (PLAN-continuous §13) agrees with the rendered pixels (no codec).

For every scenario: the viewport equals the frame size, the page background is the colour of a
pixel outside all content, each element's ``background_color`` is the median colour of its
inner area (children and rounded corners excluded), its ``text_color`` occurs at full coverage
inside its text, and the text's ink height is consistent with ``cap_height_px`` /
``font_size_px``.
"""

from __future__ import annotations

import numpy as np
import pytest

from . import scenarios as S
from .scene import Node, Renderer, Scene, hex_to_rgb, text_size

#: A pixel outside every scene's content (right of all cards/menus, below everything but the
#: footer text, which sits at x 32).
PAGE_PROBE = (1275, 795)

ALL = [e.name for e in S.SCENARIOS]
EXISTING = [e.name for e in S.TRANSITION_SCENARIOS]


@pytest.fixture(scope="module")
def built() -> dict:
    out = {}
    for e in S.SCENARIOS:
        defn = e.build()
        out[e.name] = (e, defn, S.build_truth(e, defn))
    return out


def _bgr(h: str) -> np.ndarray:
    return np.array(hex_to_rgb(h)[::-1], np.float64)


def _when(truth_el, defn) -> float | None:
    """A time at which the element is fully opaque and visible (state A preferred)."""
    ap = truth_el.appearance
    if defn.continuous is not None:
        return 0.0
    t_a, t_b = S.state_times(defn)
    if truth_el.visible_initial and ap.opacity_initial >= 1.0:
        return t_a
    if truth_el.visible_active and ap.opacity_active >= 1.0:
        return t_b
    return None


def _inner_mask(scene: Scene, node: Node, t: float, shape: tuple[int, int]) -> np.ndarray:
    """Pixels inside the node box, inset past its corner radius, minus the boxes (+2 px) of
    visible descendants (invisible layout groups don't cover anything)."""
    h, w = shape
    m = np.zeros((h, w), bool)
    bx, by, bw, bh = scene.box(node.id, t)
    inset = max(node.radius, 2.0) + 2.0
    x0, y0 = int(np.ceil(bx + inset)), int(np.ceil(by + inset))
    x1, y1 = int(np.floor(bx + bw - inset)), int(np.floor(by + bh - inset))
    m[max(y0, 0) : max(y1, 0), max(x0, 0) : max(x1, 0)] = True
    for d in node.walk():
        if d is node or d.shape == "group":
            continue
        cx, cy, cw, ch = scene.box(d.id, t)
        m[max(int(cy) - 2, 0) : int(cy + ch) + 3, max(int(cx) - 2, 0) : int(cx + cw) + 3] = False
    return m


def _texts(node: Node) -> list[Node]:
    return [n for n in node.walk() if n.shape == "text"]


@pytest.mark.parametrize("name", ALL)
def test_scene_truth(name: str, built: dict) -> None:
    entry, defn, truth = built[name]
    assert truth.scene is not None
    assert (truth.scene.viewport_css.w, truth.scene.viewport_css.h) == (S.CSS_W, S.CSS_H)
    dark = defn.continuous is not None and defn.continuous.strip.page_bg is not None
    want_bg = defn.continuous.strip.page_bg if dark else S.PAGE_BG  # type: ignore[union-attr]
    assert truth.scene.page_background == want_bg
    r = Renderer(defn.scene, entry.variant.pixel_ratio)
    f = r.frame(0.0)
    ratio = entry.variant.pixel_ratio
    assert f.shape[:2] == (S.CSS_H * ratio, S.CSS_W * ratio)
    px = f[PAGE_PROBE[1] * ratio, PAGE_PROBE[0] * ratio].astype(np.float64)
    np.testing.assert_allclose(px, _bgr(truth.scene.page_background), atol=0.5)


@pytest.mark.parametrize("name", EXISTING)
def test_existing_scenarios_have_appearance(name: str, built: dict) -> None:
    _, defn, truth = built[name]
    for el in truth.elements:
        ap = el.appearance
        assert ap is not None, f"{name}: {el.id} has no appearance truth"
        node = defn.scene.node(el.id)
        if node.shape in ("rect", "icon"):
            assert ap.background_color is not None
        if el.text_like:
            assert ap.text_color is not None and ap.font_size_px is not None


@pytest.mark.parametrize("name", ALL)
def test_appearance_matches_pixels(name: str, built: dict) -> None:
    entry, defn, truth = built[name]
    if not truth.elements:
        pytest.skip("no elements (negative scenario)")
    scene = defn.scene
    ratio = entry.variant.pixel_ratio
    r = Renderer(scene, ratio)
    checked = 0
    for el in truth.elements:
        ap = el.appearance
        assert ap is not None
        t = _when(el, defn)
        if t is None:
            continue  # e.g. the backdrop: never fully opaque
        frame = r.frame(t).astype(np.float64)
        if ratio != 1:
            frame = frame[::ratio, ::ratio]  # CSS-px grid is enough for flat colours
        node = scene.node(el.id)
        if ap.background_color is not None:
            m = _inner_mask(scene, node, t, frame.shape[:2])
            assert m.sum() > 50, f"{el.id}: no inner area to sample"
            med = np.median(frame[m], axis=0)
            np.testing.assert_allclose(med, _bgr(ap.background_color), atol=1.0,
                                       err_msg=f"{name} {el.id} background")  # fmt: skip
            checked += 1
        if ap.text_color is not None:
            want = _bgr(ap.text_color)
            for tn in _texts(node):
                bx, by, bw, bh = scene.box(tn.id, t)
                crop = frame[int(by) : int(by + bh) + 1, int(bx) : int(bx + bw) + 1]
                dist = np.abs(crop - want).max(axis=2)
                assert (dist <= 3).sum() >= 3, f"{name} {tn.id}: text colour not found"
                if ap.font_size_px is not None:
                    bg = np.median(frame[int(by) - 1, int(bx) : int(bx + bw)], axis=0)
                    ink = np.abs(crop - bg).max(axis=2) >= 0.5 * np.abs(want - bg).max()
                    rows = np.flatnonzero(ink.any(axis=1))
                    ink_h = rows[-1] - rows[0] + 1
                    _, asc, desc = text_size(tn.text, tn.font_px, tn.weight)
                    assert ap.cap_height_px is not None
                    assert ap.cap_height_px - 1 <= ink_h <= asc + desc + 2, (
                        f"{name} {tn.id}: ink height {ink_h} vs cap {ap.cap_height_px}"
                    )
                checked += 1
    assert checked > 0, f"{name}: nothing checked"


def test_cap_height_is_measured_from_the_renderer() -> None:
    # the synth font: cap height ≈ 0.75 × font size (ascent metric); grows linearly
    assert S.cap_height(20.0) == pytest.approx(15.0, abs=0.3)
    assert S.cap_height(40.0) == pytest.approx(2 * S.cap_height(20.0), abs=0.3)


def test_continuous_card_appearance_matches_pixels(built: dict) -> None:
    entry, defn, truth = built["carousel_drag_inertia"]
    tc = truth.continuous
    assert tc is not None
    scene, card = defn.scene, tc.card
    frame = Renderer(scene).frame(0.0).astype(np.float64)
    node = scene.node("c1_0")  # second copy, card 1: at the viewport start when x = 0
    bx, by, bw, bh = scene.box(node.id, 0.0)
    assert (bw, bh) == (card.w, card.h)
    assert bx == pytest.approx(tc.region.x + 8)
    m = _inner_mask(scene, node, 0.0, frame.shape[:2])
    np.testing.assert_allclose(np.median(frame[m], axis=0), _bgr(card.background_color), atol=1)
    nxt = scene.box("c1_1", 0.0)
    assert nxt[0] - bx == pytest.approx(tc.pitch_px)
    gap_px = frame[int(by + bh / 2), int(bx + bw + tc.gap_px / 2)]
    np.testing.assert_allclose(gap_px, _bgr(truth.elements[0].appearance.background_color), atol=1)


def test_dark_zoomed_cards_match_pixels(built: dict) -> None:
    """C13 (P2b): dark low-contrast cards, scaled by their distance from the viewport centre;
    the truth card (scale 1) / gap / pitch are the geometry at the centre."""
    entry, defn, truth = built["carousel_dark_edge_zoom"]
    tc = truth.continuous
    assert tc is not None and tc.zoom is not None
    strip = defn.continuous.strip  # type: ignore[union-attr]
    zoom = strip.zoom
    assert zoom is not None
    scene = defn.scene
    frame = Renderer(scene).frame(0.0).astype(np.float64)
    vx, vy, vw, vh = strip.box
    centre = vx + vw / 2.0
    sizes = []
    for k in range(strip.cards):
        for cp in (0, 1):
            cid = f"c{cp}_{k}"
            bx, by, bw, bh = scene.box(cid, 0.0, clipped=False)
            if bx < vx or bx + bw > vx + vw:
                continue  # clipped by the viewport
            d = bx + bw / 2.0 - centre
            sizes.append((abs(d), bw, bh))
            # uniform scale about the card centre, magnification 1 + a d² at its centre
            assert bw / bh == pytest.approx(strip.card_w / strip.card_h, rel=1e-6)
            assert bw / strip.card_w == pytest.approx(1.0 + zoom.a * d * d, rel=1e-6)
    assert len(sizes) >= 3
    sizes.sort()
    assert all(b[1] >= a[1] for a, b in zip(sizes, sizes[1:], strict=False))
    # the page and the track share the dark background; the card fill sits inside the border
    page = frame[5, 5]
    np.testing.assert_allclose(page, _bgr(S.DARK_PAGE), atol=0.5)
    gap, pitch = strip.centre_geometry()
    assert tc.gap_px == pytest.approx(gap, abs=1e-3)
    assert tc.pitch_px == pytest.approx(pitch, abs=1e-3)
    assert tc.card.w == strip.card_w and tc.card.background_color == S.DARK_CARD
    assert 2.5 < gap < strip.gap  # the lens' convexity narrows the gap a little at the centre
    assert tc.zoom.speed_scale == pytest.approx(zoom.mean_scale(), abs=1e-6)
