"""A4 state frames + element decomposition on rendered A/B pairs (Track M1)."""

from __future__ import annotations

import numpy as np
import pytest

from app.pipeline.params import DEFAULT_PARAMS as P
from app.pipeline.regions import (
    StateFrames,
    components,
    detect_elements,
    masked_median,
    state_frames,
    text_like,
)
from app.pipeline.track import decompose
from tests.unit.m1_helpers import (
    base_canvas,
    card_sprite,
    composite,
    scale_about,
    solid_rect_sprite,
    textured_patch,
    translate,
    window_frames,
)

W, H = 480, 320


def detect(a: np.ndarray, b: np.ndarray):
    wf = window_frames([a, b], 60.0)
    st = StateFrames(a, b, np.full(a.shape[:2], 255, np.uint8), [0], [1])
    return detect_elements(wf, st, (W, H), P)


def test_masked_median_ignores_cursor_pixels() -> None:
    frames = [np.full((10, 10, 3), 100, np.uint8) for _ in range(3)]
    frames[0][2:5, 2:5] = 0  # cursor in frame 0 …
    frames[1][2:5, 2:5] = 0  # … and frame 1 (majority!) but masked
    masks = [np.zeros((10, 10), np.uint8) for _ in range(3)]
    masks[0][2:5, 2:5] = 255
    masks[1][2:5, 2:5] = 255
    img, valid = masked_median(frames, masks)
    assert (img == 100).all() and (valid == 255).all()


def test_state_frames_pick_pre_and_post() -> None:
    frames = [np.full((8, 8, 3), 10 if i < 10 else 200, np.uint8) for i in range(30)]
    wf = window_frames(frames, 60.0)
    st = state_frames(wf, 9 / 60, 12 / 60, P)
    assert (st.a == 10).all() and (st.b == 200).all()
    assert max(st.a_idx) <= 9 and min(st.b_idx) >= 12


def test_components_merge_close_boxes() -> None:
    m = np.zeros((50, 100), bool)
    m[10:20, 10:20] = True
    m[10:20, 25:35] = True  # 5 px gap → merged
    m[10:20, 80:90] = True
    comps = components(m, 10, 12)
    assert len(comps) == 2


def test_translated_card_is_one_transform_element() -> None:
    card = card_sprite(180, 200)
    a, b = base_canvas(W, H), base_canvas(W, H)
    composite(a, card, translate(140, 70))
    composite(b, card, translate(140, 62))
    rr = detect(a, b)
    assert len(rr.elements) == 1
    e = rr.elements[0]
    assert e.kind == "transform" and e.parent_id is None
    assert e.warp_ab[1, 2] == pytest.approx(-8.0, abs=0.2)
    assert e.bbox_a.y == pytest.approx(70, abs=1.5) and e.bbox_a.h == pytest.approx(200, abs=2)


def test_fill_color_change_is_photometric() -> None:
    a, b = base_canvas(W, H), base_canvas(W, H)
    composite(a, solid_rect_sprite(160, 44, (246, 130, 59), 6), translate(160, 140))
    composite(b, solid_rect_sprite(160, 44, (216, 78, 29), 6), translate(160, 140))
    rr = detect(a, b)
    assert [e.kind for e in rr.elements] == ["photometric"]
    assert not rr.elements[0].text_like


def test_box_on_background_appears() -> None:
    a, b = base_canvas(W, H), base_canvas(W, H)
    composite(b, solid_rect_sprite(200, 120, (255, 255, 255), 8), translate(150, 100))
    b[130:140, 170:300] = (60, 60, 60)
    rr = detect(a, b)
    assert rr.elements[0].kind == "appear"
    assert rr.elements[0].bbox_b.x == pytest.approx(150, abs=3)


def test_backdrop_with_panel_child() -> None:
    a = base_canvas(W, H)
    a[40:300, 20:460] = textured_patch(440, 260, 5) // 2 + 100
    b = (a.astype(np.float32) * 0.5).astype(np.uint8)
    panel = solid_rect_sprite(200, 140, (255, 255, 255), 10)
    composite(b, panel, translate(140, 90))
    b[120:130, 160:300] = (40, 40, 40)
    rr = detect(a, b)
    kinds = {e.kind for e in rr.elements}
    assert "backdrop" in kinds
    bd = next(e for e in rr.elements if e.kind == "backdrop")
    assert rr.overlay[bd.id][0] == "#000000"
    assert rr.overlay[bd.id][1] == pytest.approx(0.5, abs=0.05)
    assert any(e.parent_id == bd.id and e.kind == "appear" for e in rr.elements)


def test_card_with_clipped_zooming_image_child() -> None:
    card = card_sprite(240, 260, seed=1)
    img = textured_patch(240, 130, seed=9)

    def render(dy: float, s: float) -> np.ndarray:
        out = base_canvas(W, H)
        composite(out, card, translate(120, 30 + dy))
        z = np.zeros((130, 240, 4), np.uint8)
        z[..., :3] = img
        z[..., 3] = 255
        import cv2

        m = scale_about(s, 120, 65)[:2]
        zz = cv2.warpAffine(z, m, (240, 130), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT)  # fmt: skip
        composite(out, zz, translate(120, 30 + dy))  # clipped to the image box
        return out

    rr = detect(render(0, 1.0), render(-8, 1.06))
    roots = [e for e in rr.elements if e.parent_id is None]
    assert len(roots) == 1 and roots[0].kind == "transform"
    assert roots[0].warp_ab[1, 2] == pytest.approx(-8, abs=0.5)
    kids = [e for e in rr.elements if e.parent_id == roots[0].id and e.kind == "transform"]
    assert kids, rr.elements
    assert decompose(kids[0].warp_ab).scale == pytest.approx(1.06, abs=0.01)


def test_text_like() -> None:
    m = np.zeros((20, 100), bool)
    for x in range(5, 95, 12):  # glyph-like strokes
        m[4:16, x : x + 2] = True
    assert text_like(m, (0, 0, 100, 20), 1.0, P)
    full = np.ones((40, 100), bool)
    assert not text_like(full, (0, 0, 100, 40), 1.0, P)


def test_expand_collapse_resize_band() -> None:
    import cv2

    def row(img: np.ndarray, y: int, label: str) -> None:
        composite(img, solid_rect_sprite(300, 40, (252, 252, 252), 6), translate(80, y))
        cv2.putText(img, label, (96, y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1,
                    cv2.LINE_AA)  # fmt: skip

    a, b = base_canvas(W, H), base_canvas(W, H)
    for img, shift in ((a, 0), (b, 60)):
        row(img, 40, "Shipping")
        row(img, 90 + shift, "Returns")
        row(img, 140 + shift, "Warranty")
    for i, txt in enumerate(("Ships in 2 days.", "Free over $50.")):
        cv2.putText(b, txt, (96, 106 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1,
                    cv2.LINE_AA)  # fmt: skip
    rr = detect(a, b)
    resize = [e for e in rr.elements if e.kind == "resize"]
    assert len(resize) == 1
    pushed = rr.resize_links[resize[0].id]
    assert pushed
    for pid in pushed:
        e = next(x for x in rr.elements if x.id == pid)
        assert "caused_by_resize" in e.notes
        assert e.warp_ab[1, 2] == pytest.approx(60, abs=0.5)
    assert resize[0].bbox_b.h == pytest.approx(60, abs=1) and resize[0].bbox_a.h == 0


def test_nanmedian_axis0_matches_numpy() -> None:
    import warnings

    from app.pipeline.regions import nanmedian_axis0

    rng = np.random.default_rng(3)
    for n in (1, 2, 5, 8, 9):
        s = rng.integers(0, 256, (n, 20, 30, 3)).astype(np.float32)
        s[rng.random((n, 20, 30)) < 0.45] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            ref = np.nanmedian(s, axis=0)
        got = nanmedian_axis0(s)
        assert got.dtype == np.float32
        assert np.array_equal(ref, got, equal_nan=True)
